from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from trialmatchai.config.config_loader import load_config
from trialmatchai.entities import build_entity_annotator
from trialmatchai.matching.eligibility_reasoning import BatchTrialProcessor
from trialmatchai.matching.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever
from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.services.preflight import run_preflight_checks
from trialmatchai.interop.exporters import profile_to_matching_summary
from trialmatchai.interop.models import PatientProfile
from trialmatchai.utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from trialmatchai.schemas.phenopacket import Keywords
from trialmatchai.utils.logging_config import reset_request_id, set_request_id, setup_logging
from trialmatchai.utils.timing import log_timing

if TYPE_CHECKING:
    from trialmatchai.models.embedding.text_embedder import TextEmbedder

logger = setup_logging(__name__)


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    entity_annotator,
    embedder: TextEmbedder,
    config: Dict,
    search_backend,
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

    cts = ClinicalTrialSearch(
        search_backend=search_backend,
        embedder=embedder,
        entity_annotator=entity_annotator,
    )

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
        from trialmatchai.matching.eligibility_reasoning_vllm import (
            BatchTrialProcessorVLLM,
        )
        from trialmatchai.models.llm.vllm_loader import load_vllm_engine

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

        batch_size = min(config["rag"]["batch_size"] * 2, 8)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rag_processor = BatchTrialProcessor(
            model,
            tokenizer,
            device=config["global"]["device"],
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


def main_pipeline(config_path: str | None = None) -> int:
    logger.info("Starting TrialMatchAI pipeline...")
    config = load_config(config_path)
    paths = config["paths"]
    create_directory(paths["output_dir"])

    search_backend = LanceDBSearchBackend.from_config(config)
    preflight_issues = run_preflight_checks(
        config,
        require_patient_inputs=True,
        require_trials_json=True,
        require_models=True,
        require_search_tables=False,
    )
    if preflight_issues:
        return 1

    index_issues = run_preflight_checks(
        config,
        search_backend=search_backend,
        require_search_tables=True,
    )
    if index_issues:
        return 1

    patient_inputs = _load_patient_inputs(config)
    if not patient_inputs:
        logger.error("No patient profiles were available for matching.")
        return 1

    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch is required to run matching. Install the ML extras with "
            "`uv sync --extra llm --extra entity`."
        )
        return 1

    from trialmatchai.models.embedding import build_embedder
    from trialmatchai.models.llm.llm_loader import load_model_and_tokenizer
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    import warnings

    # The HuggingFace CoT model is only used by the default backend. When the
    # vLLM backend is configured, run_rag_processing loads its own engine and
    # ignores these, so skip loading (and half()-ing) the HF model entirely to
    # avoid wasting GPU memory and load time (and risking OOM next to vLLM).
    model, tokenizer = None, None
    if config.get("cot_backend", "default") != "vllm":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*quantization_config.*", category=UserWarning
            )
            model, tokenizer = load_model_and_tokenizer(
                config["model"], config["global"]["device"]
            )

        if tokenizer.pad_token is None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:  # type: ignore
                model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

        if config["global"]["device"] != "cpu" and torch.cuda.is_available():
            model = model.half()  # type: ignore

    # Initialize components
    embedder = build_embedder(config)
    entity_annotator = build_entity_annotator(config, embedder=embedder)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        llm_reranker = LLMReranker(
            model_path=config["model"]["reranker_model_path"],
            adapter_path=config["model"]["reranker_adapter_path"],
            device=config["global"]["device"],
            batch_size=config["rag"]["batch_size"] * 2,
            revision=config["model"].get("reranker_model_revision"),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )

    gemma_retriever = SecondStageRetriever(
        search_backend=search_backend,
        llm_reranker=llm_reranker,
        embedder=embedder,
        entity_annotator=entity_annotator,
        search_mode=config["search"].get("mode", "hybrid"),
    )

    for profile, summary in patient_inputs:
        patient_id = profile.patient_id
        token = set_request_id(patient_id)
        output_folder = Path(paths["output_dir"]) / patient_id
        create_directory(str(output_folder))

        try:
            write_json_file(summary, str(output_folder / "keywords.json"))
            write_json_file(
                profile.model_dump(mode="json", exclude_none=True),
                str(output_folder / "patient_profile.json"),
            )
            keywords = Keywords.model_validate(summary).model_dump()
            patient_info = dict(summary)
            patient_info["split_raw_description"] = summary.get(
                "expanded_sentences", []
            )

            # Run pipeline
            with log_timing(logger, "First-level search"):
                with torch.no_grad():
                    result = run_first_level_search(
                        keywords,
                        str(output_folder),
                        patient_info,
                        entity_annotator,
                        embedder,
                        config,
                        search_backend,
                    )
            if not result:
                logger.error("First-level search failed for %s", patient_id)
                continue

            (
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                first_level_scores,
            ) = result

            with log_timing(logger, "Second-level search"):
                with torch.no_grad():
                    _, top_trials_path = run_second_level_search(
                        str(output_folder),
                        nct_ids,
                        main_conditions,
                        other_conditions,
                        expanded_sentences,
                        gemma_retriever,
                        first_level_scores,
                        config,
                    )

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

            with log_timing(logger, "Final ranking"):
                trial_data = load_trial_data(str(output_folder))
                ranked_trials = rank_trials(trial_data)
                save_ranked_trials(
                    ranked_trials, str(output_folder / "ranked_trials.json")
                )

            logger.info("Pipeline completed for patient %s", patient_id)
        except Exception:
            logger.exception("Pipeline failed for patient %s", patient_id)
            continue
        finally:
            reset_request_id(token)

    return 0


def _load_patient_inputs(config: Dict) -> list[tuple[PatientProfile, Dict]]:
    patient_cfg = config.get("patient_inputs", {})
    profile_dir = Path(patient_cfg.get("profile_dir", "data/patients/profiles"))
    summary_dir = Path(patient_cfg.get("summary_dir", "data/patients/summaries"))
    profile_files = sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []
    if not profile_files:
        logger.error(
            "No canonical patient profiles found in %s. Run "
            "`trialmatchai-import-patient` first.",
            profile_dir,
        )
        return []

    loaded: list[tuple[PatientProfile, Dict]] = []
    for profile_file in profile_files:
        try:
            profile = PatientProfile.model_validate(read_json_file(str(profile_file)))
            summary_path = summary_dir / profile_file.name
            if summary_path.exists():
                summary = read_json_file(str(summary_path))
            else:
                summary = profile_to_matching_summary(profile)
            loaded.append((profile, summary))
        except Exception:
            logger.exception("Failed to load patient profile: %s", profile_file)
    return loaded


if __name__ == "__main__":
    raise SystemExit(main_pipeline())
