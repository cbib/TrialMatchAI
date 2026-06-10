from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from Parser.biomedner_engine import BioMedNER

from elasticsearch import Elasticsearch

from Matcher.config.config_loader import load_config
from Matcher.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig
from Matcher.models.llm.llm_loader import load_model_and_tokenizer
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.models.llm.vllm_loader import load_vllm_engine
from Matcher.pipeline.cot_reasoning import BatchTrialProcessor
from Matcher.pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM
from Matcher.pipeline.phenopacket_processor import PhenopacketProcessor, process_phenopacket
from Matcher.pipeline.genomic_filter import extract_genomic_terms
from Matcher.pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch
from Matcher.pipeline.trial_search.second_level_search import SecondStageRetriever
from Matcher.services.biomedner_service import initialize_biomedner_services
from Matcher.services.elasticsearch_service import ensure_elasticsearch
from Matcher.utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from Matcher.schemas.phenopacket import Keywords, Phenopacket
from Matcher.utils.logging_config import reset_request_id, set_request_id, setup_logging
from Matcher.utils.timing import log_timing

logger = setup_logging(__name__)


def extract_keywords_without_llm(phenopacket_path: str) -> Dict:
    """Build Keywords dict from phenopacket structured fields, no LLM required."""
    data = read_json_file(phenopacket_path)
    processor = PhenopacketProcessor(phenopacket_path)
    sentences = processor.generate_medical_narrative()

    main_conditions = [
        d.get("term", {}).get("label", "")
        for d in data.get("diseases", [])
        if d.get("term", {}).get("label")
    ]
    other_conditions = [
        pf.get("type", {}).get("label", "")
        for pf in data.get("phenotypicFeatures", [])
        if not pf.get("excluded", False) and pf.get("type", {}).get("label")
    ]
    return {
        "main_conditions": main_conditions,
        "other_conditions": other_conditions,
        "expanded_sentences": sentences,
    }


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: TextEmbedder,
    config: Dict,
    es_client: Elasticsearch,
) -> Optional[Tuple]:
    main_conditions = keywords.get("main_conditions", [])
    other_conditions = keywords.get("other_conditions", [])
    expanded_sentences = keywords.get("expanded_sentences", [])

    if not main_conditions:
        logger.error("No main_conditions found in keywords.")
        return None

    condition = main_conditions[0]
    age = patient_info.get("age", "all")
    sex = patient_info.get("gender", "all")
    overall_status = "All"

    index_name = config["elasticsearch"]["index_trials"]
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    # Get synonyms and expand main conditions
    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[:5])

    search_size = config["search"].get("max_trials_first_level", 300)
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=search_size,
        pre_selected_nct_ids=None,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=config["search"]["vector_score_threshold"],
    )

    nct_ids = [trial.get("nct_id") for trial in trials if trial.get("nct_id")]
    first_level_scores = {
        trial.get("nct_id"): score
        for trial, score in zip(trials, scores)
        if trial.get("nct_id")
    }

    write_text_file([str(nid) for nid in nct_ids], f"{output_folder}/nct_ids.txt")
    write_json_file(first_level_scores, f"{output_folder}/first_level_scores.json")

    logger.info(f"First-level search complete: {len(nct_ids)} trial IDs saved.")
    return (
        nct_ids,
        main_conditions,
        other_conditions,
        expanded_sentences,
        first_level_scores,
    )


def run_second_level_search(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    expanded_sentences: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
    config: Dict,
) -> Tuple:
    queries = list(set(main_conditions + other_conditions + expanded_sentences))[:10]
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")

    # Add synonyms for second level
    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    top_n = min(len(nct_ids), config["search"].get("max_trials_second_level", 100))
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=top_n
    )

    combined_scores = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = trial["score"]
        first_score = first_level_scores.get(trial_id, 0)
        combined_scores[trial_id] = first_score + second_score

    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, min(len(sorted_trials) // 3, top_n))
    semi_final_trials = sorted_trials[:num_top]

    top_trials_path = f"{output_folder}/top_trials.txt"
    write_text_file([trial_id for trial_id, _ in semi_final_trials], top_trials_path)

    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


def run_rag_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    model,
    tokenizer,
    config: Dict,
):
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return

    top_trials = top_trials[: config["rag"].get("max_trials_rag", 20)]
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return

    # Check if vLLM backend is configured
    cot_backend = config.get("cot_backend", "default")
    use_vllm = cot_backend == "vllm"

    if use_vllm:
        logger.info("Using vLLM backend for CoT reasoning")

        # Load vLLM configuration
        vllm_cfg = config.get("vllm", {})

        # Load vLLM engine
        vllm_engine, vllm_tokenizer, lora_request = load_vllm_engine(
            model_config=config.get("model", {}),
            vllm_cfg=vllm_cfg,
        )

        # Create vLLM processor
        rag_processor = BatchTrialProcessorVLLM(
            llm=vllm_engine,  # type: ignore
            tokenizer=vllm_tokenizer,
            batch_size=vllm_cfg.get("batch_size", 16),
            use_cot=config.get("use_cot_reasoning", True),
            max_new_tokens=vllm_cfg.get("max_new_tokens", 5000),
            temperature=vllm_cfg.get("temperature", 0.0),
            top_p=vllm_cfg.get("top_p", 1.0),
            seed=vllm_cfg.get("seed", 1234),
            length_bucket=vllm_cfg.get("length_bucket", True),
            lora_request=lora_request,
        )
    else:
        logger.info("Using default (HuggingFace) backend for CoT reasoning")

        # MPS can't run phi-4 in parallel batches without OOM; CUDA can
        if torch.backends.mps.is_available():
            batch_size = 1
        else:
            batch_size = min(config["rag"]["batch_size"] * 2, 8)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Resolve device: use MPS string on Apple Silicon, int on CUDA, "cpu" otherwise
        if torch.cuda.is_available():
            cot_device = config["global"]["device"]
        elif torch.backends.mps.is_available():
            cot_device = "mps"
        else:
            cot_device = "cpu"

        rag_processor = BatchTrialProcessor(
            model,
            tokenizer,
            device=cot_device,
            batch_size=batch_size,
        )

    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )
    write_json_file({"status": "done"}, f"{output_folder}/rag_output.json")
    logger.info("RAG-based trial matching complete.")


def step(n: str, msg: str):
    logger.info("── [Step %s] %s", n, msg)


def main_pipeline(config_path: str = "Matcher/config/config.json", skip_llm: bool = False):
    mode = "retrieval-only (--skip-llm)" if skip_llm else "full pipeline (CoT + reranker)"
    logger.info("=" * 60)
    logger.info("  TrialMatchAI  |  mode: %s", mode)
    logger.info("=" * 60)

    step("1.0", "Loading configuration")
    config = load_config(config_path)
    paths = config["paths"]
    create_directory(paths["output_dir"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    step("1.1", "Starting BioMedNER services (NER + Java normalizers)")
    initialize_biomedner_services(config)

    import warnings

    model, tokenizer = None, None
    if not skip_llm:
        step("1.2", f"Loading CoT model: {config['model']['base_model']}  (this takes ~2–3 min on MPS)")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*quantization_config.*", category=UserWarning
            )
            model, tokenizer = load_model_and_tokenizer(
                config["model"], config["global"]["device"]
            )
        logger.info("    ✓ CoT model loaded")

        if tokenizer.pad_token is None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:  # type: ignore
                model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

        # half() is only needed on CUDA when the loader didn't already set fp16/bf16
        if torch.cuda.is_available() and config["global"]["device"] != "cpu":
            model = model.half()  # type: ignore

    step("1.3", f"Loading embedder: {config.get('embedder', {}).get('model_name', 'BAAI/bge-m3')}")
    embedder_cfg = config.get("embedder", {})
    embedder = TextEmbedder(
        TextEmbedderConfig(
            model_name=embedder_cfg.get("model_name", "BAAI/bge-m3"),
            pooling=embedder_cfg.get("pooling", "mean"),
            max_length=embedder_cfg.get("max_length", 512),
            batch_size=embedder_cfg.get("batch_size", 32),
            use_gpu=embedder_cfg.get("use_gpu", True),
            use_fp16=embedder_cfg.get("use_fp16", False),
            normalize=embedder_cfg.get("normalize", True),
        )
    )
    logger.info("    ✓ Embedder ready")

    step("1.4", "Initializing BioMedNER client")
    try:
        bio_med_ner = BioMedNER(**config["bio_med_ner"])
        logger.info("    ✓ BioMedNER client ready")
    except Exception:
        logger.warning("    ✗ BioMedNER failed to initialize; synonym expansion will be disabled.", exc_info=True)
        bio_med_ner = None

    llm_reranker = None
    if not skip_llm:
        step("1.5", f"Loading LLM reranker: {config['model']['reranker_model_path']}")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*quantization_config.*", category=UserWarning
            )
            llm_reranker = LLMReranker(
                model_path=config["model"]["reranker_model_path"],
                adapter_path=config["model"]["reranker_adapter_path"],
                device=config["global"]["device"],
                batch_size=config["rag"]["batch_size"] * 2,
            )
        logger.info("    ✓ LLM reranker ready")

    step("1.6", "Connecting to Elasticsearch")
    es_client = Elasticsearch(
        hosts=[config["elasticsearch"]["host"]],
        ca_certs=paths["docker_certs"],
        basic_auth=(
            config["elasticsearch"]["username"],
            config["elasticsearch"]["password"],
        ),
        request_timeout=config["elasticsearch"]["request_timeout"],
        retry_on_timeout=config["elasticsearch"]["retry_on_timeout"],
    )
    if not ensure_elasticsearch(es_client, config):
        return
    logger.info("    ✓ Elasticsearch connected")

    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
    )

    # Process phenopackets
    patient_folder = Path(paths["patients_dir"])
    if not patient_folder.exists():
        logger.error("Patients folder not found: %s", patient_folder)
        return
    phenopacket_files = sorted(
        [p for p in patient_folder.iterdir() if p.suffix == ".json"]
    )
    if not phenopacket_files:
        logger.warning("No patient files found in %s", patient_folder)
        return

    logger.info("─" * 60)
    logger.info("  Found %d patient file(s) to process", len(phenopacket_files))

    for i, phenopacket_path in enumerate(phenopacket_files, 1):
        patient_id = phenopacket_path.stem
        token = set_request_id(patient_id)
        output_folder = Path(paths["output_dir"]) / patient_id
        create_directory(str(output_folder))

        logger.info("─" * 60)
        logger.info("  Patient %d/%d: %s", i, len(phenopacket_files), patient_id)

        input_file = str(phenopacket_path)
        output_file = str(output_folder / "keywords.json")

        try:
            step("2.1", "Extracting keywords from phenopacket")
            with log_timing(logger, "Phenopacket processing"):
                if skip_llm:
                    keywords = extract_keywords_without_llm(input_file)
                    write_json_file(keywords, output_file)
                    logger.info("    ✓ Keywords extracted (structure-based, no LLM)")
                else:
                    with torch.no_grad():
                        process_phenopacket(
                            input_file, output_file, model=model, tokenizer=tokenizer
                        )
                    keywords = Keywords.model_validate(read_json_file(output_file)).model_dump()
                    logger.info("    ✓ Keywords extracted via phi-4 CoT")

            patient_info = Phenopacket.model_validate(
                read_json_file(input_file)
            ).model_dump()
            patient_info["split_raw_description"] = keywords.get(
                "expanded_sentences", []
            )
            logger.info("    Queries to run: %d", len(keywords.get("expanded_sentences", [])))

            genomic_terms = extract_genomic_terms(read_json_file(input_file))
            if genomic_terms:
                keywords["other_conditions"] = genomic_terms + keywords.get("other_conditions", [])
                logger.info("    ✓ Injected %d genomic variant term(s): %s", len(genomic_terms), genomic_terms)
            else:
                logger.info("    (no genomicInterpretations found in phenopacket)")

            step("2.2", "First-level search — BM25 + vector over clinical_trials index")
            with log_timing(logger, "First-level search"):
                with torch.no_grad():
                    result = run_first_level_search(
                        keywords,
                        str(output_folder),
                        patient_info,
                        bio_med_ner,
                        embedder,
                        config,
                        es_client,
                    )
            if not result:
                logger.error("    ✗ First-level search failed for %s", patient_id)
                continue

            (
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                first_level_scores,
            ) = result
            logger.info("    ✓ Retrieved %d candidate trials", len(nct_ids))

            step("2.3", f"Second-level retrieval — hybrid search over trials_eligibility index ({len(nct_ids)} candidates)")
            with log_timing(logger, "Second-level search"):
                with torch.no_grad():
                    semi_final_trials, top_trials_path = run_second_level_search(
                        str(output_folder),
                        nct_ids,
                        main_conditions,
                        other_conditions,
                        expanded_sentences,
                        gemma_retriever,
                        first_level_scores,
                        config,
                    )
            logger.info("    ✓ Shortlisted %d trials for ranking", len(semi_final_trials))

            if skip_llm:
                ranked_trials = [
                    {"TrialID": nct_id, "Score": score}
                    for nct_id, score in semi_final_trials
                ]
                save_ranked_trials(
                    ranked_trials, str(output_folder / "ranked_trials.json")
                )
                logger.info("─" * 60)
                logger.info("  ✓ Retrieval-only pipeline complete — %d trials ranked", len(ranked_trials))
                logger.info("    Output: %s/ranked_trials.json", output_folder)
            else:
                step("2.4", f"CoT reasoning — evaluating eligibility criteria for top trials (expect ~30–60s/trial on MPS)")
                with log_timing(logger, "RAG processing"):
                    with torch.no_grad():
                        run_rag_processing(
                            str(output_folder),
                            top_trials_path,
                            patient_info,
                            model,
                            tokenizer,
                            config,
                        )
                logger.info("    ✓ CoT evaluation complete")

                step("2.5", "Final ranking — aggregating CoT scores")
                with log_timing(logger, "Final ranking"):
                    trial_data = load_trial_data(str(output_folder))
                    ranked_trials = rank_trials(trial_data)
                    save_ranked_trials(
                        ranked_trials, str(output_folder / "ranked_trials.json")
                    )

                logger.info("─" * 60)
                logger.info("  ✓ Pipeline complete for %s — %d trials ranked", patient_id, len(ranked_trials))
                logger.info("    Output: %s/ranked_trials.json", output_folder)
        except Exception:
            logger.exception("Pipeline failed for patient %s", patient_id)
            continue
        finally:
            reset_request_id(token)

    logger.info("=" * 60)
    logger.info("  All patients processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrialMatchAI pipeline")
    parser.add_argument("--config", default="Matcher/config/config.json", help="Path to config.json")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM model loading and RAG/CoT stages; run BM25+vector retrieval only. "
             "Useful for local testing on machines without a GPU.",
    )
    args = parser.parse_args()
    main_pipeline(args.config, skip_llm=args.skip_llm)
