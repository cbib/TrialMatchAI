from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from trialmatchai.config.config_loader import load_config
from trialmatchai.constraints import (
    build_patient_constraint_context,
    write_constraint_reports,
)
from trialmatchai.entities import build_entity_annotator
from trialmatchai.matching.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever
from trialmatchai.matching.retrieval.location import (
    filter_trials_by_country,
    patient_country,
)
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
    from trialmatchai.constraints import PatientConstraintContext
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
    patient_profile: PatientProfile | None = None,
) -> Optional[Tuple]:
    main_conditions = list(keywords.get("main_conditions", []))
    other_conditions = list(keywords.get("other_conditions", []))
    patient_narrative = list(keywords.get("patient_narrative", []))

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

    search_cfg = config["search"]
    first_level_cfg = _first_level_search_config(search_cfg)
    if first_level_cfg.get("enabled", True) and patient_profile is not None:
        plan = cts.build_query_plan(
            profile=patient_profile,
            matching_summary=keywords,
            config=first_level_cfg,
            age=age,
            sex=sex,
            overall_status=overall_status,
        )
        trials, scores, candidate_evidence = cts.search_trials_with_plan(
            query_plan=plan,
            age_input=age,
            sex=sex,
            overall_status=overall_status,
            size=int(first_level_cfg.get("max_trials", 1000)),
            per_channel_size=int(first_level_cfg.get("per_channel_size", 300)),
            pre_selected_nct_ids=None,
            vector_score_threshold=float(
                first_level_cfg.get("vector_score_threshold", 0.0)
            ),
            search_mode=search_cfg.get("mode", "hybrid"),
            rrf_k=int(first_level_cfg.get("rrf_k", 60)),
        )
        main_conditions = _dedupe_strings(
            [
                *main_conditions,
                *plan.terms_for("primary_condition", "concept_synonym"),
            ]
        )
        other_conditions = _dedupe_strings(
            [
                *other_conditions,
                *plan.terms_for("biomarker", "therapy", "broader_disease"),
            ]
        )
        if first_level_cfg.get("write_reports", True):
            write_json_file(
                plan.model_dump(mode="json"),
                f"{output_folder}/first_level_query_plan.json",
            )
            write_json_file(
                {
                    "candidates": [
                        item.model_dump(mode="json") for item in candidate_evidence
                    ]
                },
                f"{output_folder}/first_level_candidates.json",
            )
    else:
        # Single-query first-level retrieval path. This remains available for exact
        # behavior preservation when search.first_level.enabled=false.
        synonyms = cts.get_synonyms(condition.lower().strip())
        main_conditions.extend(synonyms[:5])

        search_size = search_cfg.get("max_trials_first_level", 300)
        trials, scores = cts.search_trials(
            condition=condition,
            age_input=age,
            sex=sex,
            overall_status=overall_status,
            size=search_size,
            pre_selected_nct_ids=None,
            synonyms=main_conditions,
            other_conditions=other_conditions,
            vector_score_threshold=search_cfg["vector_score_threshold"],
            search_mode=search_cfg.get("mode", "hybrid"),
        )

    # Optional geographic hard filter (country-level, site-aware, opt-in via
    # search.first_level.hard_filters). Recall-safe: only drops trials whose
    # known sites exclude the patient's country.
    hard_filters = first_level_cfg.get("hard_filters") or ["age", "sex", "overall_status"]
    if "location" in hard_filters and patient_profile is not None:
        country = patient_country(patient_profile)
        if country:
            before = len(trials)
            trials, scores = filter_trials_by_country(trials, scores, country)
            logger.info(
                "Location filter (country=%s): %d -> %d trials.",
                country,
                before,
                len(trials),
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
        patient_narrative,
        first_level_scores,
    )


def run_second_level_search(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    patient_narrative: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
    config: Dict,
    patient_context: Optional["PatientConstraintContext"] = None,
) -> Tuple:
    queries = list(set(main_conditions + other_conditions + patient_narrative))[:10]
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")

    # Add synonyms for second level
    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    top_n = min(len(nct_ids), config["search"].get("max_trials_second_level", 100))
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries,
        nct_ids,
        top_n=top_n,
        patient_context=patient_context,
        constraints_config=config.get("constraints", {}),
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
    constraints_config = config.get("constraints", {})
    if constraints_config.get("enabled", True) and constraints_config.get(
        "write_reports",
        True,
    ):
        write_constraint_reports(
            output_folder=output_folder,
            evaluations=gemma_retriever.last_constraint_evaluations,
            top_trials=[
                {"nct_id": trial_id, "score": score}
                for trial_id, score in semi_final_trials
            ],
        )

    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


def run_rag_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    config: Dict,
):
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return

    top_trials = top_trials[: config["rag"].get("max_trials_rag", 20)]
    patient_narrative = patient_info.get("patient_narrative", [])
    if not patient_narrative:
        logger.error("No patient narrative available for RAG processing.")
        return

    # vLLM is the only LLM backend. A configured cot_adapter_path is served as a
    # LoRA adapter (LoRARequest) by the engine loader.
    from trialmatchai.matching.eligibility_reasoning_vllm import BatchTrialProcessorVLLM
    from trialmatchai.models.llm.vllm_loader import load_vllm_engine

    vllm_cfg = config.get("vllm", {})
    vllm_engine, vllm_tokenizer, lora_request = load_vllm_engine(
        model_config=config.get("model", {}),
        vllm_cfg=vllm_cfg,
    )
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

    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_narrative=patient_narrative,
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
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    import warnings

    # The CoT reasoning model is loaded lazily by run_rag_processing as a vLLM
    # engine (the only LLM backend), so nothing to load here.

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
            patient_info["patient_narrative"] = summary.get("patient_narrative", [])
            patient_context = build_patient_constraint_context(profile)

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
                        patient_profile=profile,
                    )
            if not result:
                logger.error("First-level search failed for %s", patient_id)
                continue

            (
                nct_ids,
                main_conditions,
                other_conditions,
                patient_narrative,
                first_level_scores,
            ) = result

            with log_timing(logger, "Second-level search"):
                with torch.no_grad():
                    _, top_trials_path = run_second_level_search(
                        str(output_folder),
                        nct_ids,
                        main_conditions,
                        other_conditions,
                        patient_narrative,
                        gemma_retriever,
                        first_level_scores,
                        config,
                        patient_context,
                    )

            with log_timing(logger, "RAG processing"):
                with torch.no_grad():
                    run_rag_processing(
                        str(output_folder),
                        top_trials_path,
                        patient_info,
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


def _first_level_search_config(search_cfg: Dict) -> Dict:
    first_level_cfg = dict(search_cfg.get("first_level") or {})
    first_level_cfg.setdefault(
        "max_trials",
        search_cfg.get("max_trials_first_level", 1000),
    )
    first_level_cfg.setdefault("per_channel_size", 300)
    first_level_cfg.setdefault("rrf_k", 60)
    first_level_cfg.setdefault("vector_score_threshold", 0.0)
    first_level_cfg.setdefault("enabled", True)
    first_level_cfg.setdefault("write_reports", True)
    first_level_cfg.setdefault("llm_expansion_enabled", False)
    first_level_cfg.setdefault("llm_max_terms", 12)
    first_level_cfg.setdefault("hard_filters", ["age", "sex", "overall_status"])
    return first_level_cfg


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


if __name__ == "__main__":
    raise SystemExit(main_pipeline())
