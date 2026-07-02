from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EntityExtractionSettings(BaseModel):
    backend: Literal["gliner2", "regex", "disabled"] = "gliner2"
    model_name: str = "fastino/gliner2-base-v1"
    model_revision: str | None = None
    schema_path: str = "entity_schemas/trialmatchai.yaml"
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    batch_size: int = Field(8, ge=1)
    device: str = "auto"
    trust_remote_code: bool = False
    # Augment model NER with the deterministic genetic-variant recognizer.
    variant_regex: bool = True
    model_config = ConfigDict(extra="forbid")


class ConceptLinkerSettings(BaseModel):
    enabled: bool = True
    db_path: str = "data/concepts"
    table: str = "concepts"
    # Thresholds gate on an absolute lexical match-quality signal, not the top-normalized RRF rank.
    accept_threshold: float = Field(0.7, ge=0.0, le=1.0)
    reject_threshold: float = Field(0.5, ge=0.0, le=1.0)
    margin: float = Field(0.05, ge=0.0, le=1.0)
    rerank: Literal["none", "lexical"] = "lexical"
    search_limit: int = Field(10, ge=1)

    @field_validator("reject_threshold")
    @classmethod
    def validate_reject_threshold(cls, value: float, info):
        accept = info.data.get("accept_threshold")
        if accept is not None and value > accept:
            raise ValueError("concept_linker.reject_threshold must be <= accept_threshold")
        return value


class PathsSettings(BaseModel):
    output_dir: str
    trials_json_folder: str


class PatientInputSettings(BaseModel):
    raw_dir: str = "data/patients/raw"
    profile_dir: str = "data/patients/profiles"
    summary_dir: str = "data/patients/summaries"
    default_format: Literal["auto", "text", "phenopacket", "fhir", "fhir-ndjson", "omop"] = "auto"
    strict_validation: bool = False
    copy_raw: bool = True


class ModelQuantizationSettings(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"


class ModelSettings(BaseModel):
    base_model: str
    quantization: ModelQuantizationSettings
    cot_adapter_path: str
    reranker_model_path: str
    reranker_adapter_path: str
    trust_remote_code: bool = False
    base_model_revision: str | None = None
    reranker_model_revision: str | None = None


class TokenizerSettings(BaseModel):
    use_fast: bool = True
    padding_side: str = "left"


class GlobalSettings(BaseModel):
    device: int | str


class SearchBackendSettings(BaseModel):
    backend: Literal["lancedb"] = "lancedb"
    db_path: str = "data/search"
    trials_table: str = "trials"
    criteria_table: str = "criteria"
    candidate_limit: int = Field(1000, ge=1)


class RegistrySettings(BaseModel):
    source: Literal["clinicaltrials.gov"] = "clinicaltrials.gov"
    api_base_url: str = "https://clinicaltrials.gov/api/v2/studies"
    keywords_file: str | None = None
    since_days: int = Field(7, ge=0)
    max_studies: int | None = Field(default=None, ge=1)
    request_timeout: float = Field(30.0, gt=0)
    rate_limit_per_second: float = Field(2.0, gt=0)
    raw_dir: str = "data/registry/raw"
    manifest_path: str = "data/registry/manifest.jsonl"
    reports_dir: str = "data/registry/runs"
    failure_threshold: float = Field(0.25, ge=0.0, le=1.0)


class EmbedderSettings(BaseModel):
    backend: Literal["hf", "hashing"] = "hf"
    model_name: str = "BAAI/bge-m3"
    revision: str | None = None
    trust_remote_code: bool = False
    pooling: str = "mean"
    max_length: int = Field(512, ge=1)
    batch_size: int = Field(32, ge=1)
    use_gpu: bool = True
    use_fp16: bool = False
    normalize: bool = True
    hashing_dimensions: int = Field(64, ge=1)

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, value: str) -> str:
        if value not in {"mean", "cls"}:
            raise ValueError("embedder.pooling must be 'mean' or 'cls'")
        return value


class FirstLevelSearchSettings(BaseModel):
    enabled: bool = True
    max_trials: int = Field(1000, ge=1)
    per_channel_size: int = Field(300, ge=1)
    fusion: Literal["rrf"] = "rrf"
    rrf_k: int = Field(60, ge=1)
    vector_score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    llm_expansion_enabled: bool = False
    llm_max_terms: int = Field(12, ge=0)
    write_reports: bool = True
    # "location" is opt-in (country-level, site-aware); not in the default set.
    hard_filters: list[Literal["age", "sex", "overall_status", "location"]] = Field(
        default_factory=lambda: ["age", "sex", "overall_status"]
    )


class SearchSettings(BaseModel):
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    vector_score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_trials_first_level: int = Field(1000, ge=1)
    max_trials_second_level: int = Field(100, ge=1)
    # Keep the top 1/N of reranked second-level trials before CoT (N=1 keeps all).
    second_level_keep_divisor: int = Field(3, ge=1)
    first_level: FirstLevelSearchSettings = Field(
        default_factory=FirstLevelSearchSettings
    )

    @model_validator(mode="before")
    @classmethod
    def sync_first_level_alias(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        first_level = dict(normalized.get("first_level") or {})
        # Single source of truth for max_trials: nested first_level.max_trials wins when set,
        # else the top-level propagates down into it.
        nested = first_level.get("max_trials")
        top = normalized.get("max_trials_first_level")
        resolved = nested if nested is not None else top
        if resolved is not None:
            normalized["max_trials_first_level"] = resolved
            first_level["max_trials"] = resolved
            normalized["first_level"] = first_level
        return normalized


class ConstraintSettings(BaseModel):
    enabled: bool = True
    score_weight: float = Field(0.25, ge=0.0, le=1.0)
    llm_extraction_enabled: bool = False
    unknown_is_neutral: bool = True
    write_reports: bool = True


class RagSettings(BaseModel):
    enabled: bool = True
    backend: Literal["vllm", "transformers"] = "vllm"
    batch_size: int = Field(4, ge=1)
    max_trials_rag: int = Field(20, ge=1)


class VllmSettings(BaseModel):
    batch_size: int = Field(100, ge=1)
    max_new_tokens: int = Field(5000, ge=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)  # vLLM rejects top_p=0
    seed: int = 1234
    length_bucket: bool = True
    gpu_memory_utilization: float = Field(0.5, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=256)
    tensor_parallel_size: int = Field(1, ge=1)


class CotSettings(BaseModel):
    batch_size: int = Field(10, ge=1)


class LLMRerankerSettings(BaseModel):
    enabled: bool = True
    backend: Literal["vllm", "transformers"] = "vllm"
    batch_size: int = Field(20, ge=1)
    # vLLM reranker engine's share of GPU memory; lower it (with vllm.gpu_memory_utilization)
    # to fit both engines on a smaller card (e.g. 48GB A40/L40).
    gpu_memory_utilization: float = Field(0.4, gt=0.0, le=1.0)


class QueryExpansionSettings(BaseModel):
    """Runtime CoT query expansion (legacy keywords.json behaviour)."""

    enabled: bool = False
    backend: Literal["vllm", "transformers"] | None = None
    model: str | None = None
    adapter: str | None = None
    max_new_tokens: int = Field(2048, ge=1)
    max_main_conditions: int = Field(11, ge=1)
    max_other_conditions: int = Field(50, ge=1)
    trust_remote_code: bool = False


class ReportingSettings(BaseModel):
    """HTML match-report generation."""

    emit_html: bool = True


class TrialMatchSettings(BaseModel):
    entity_extraction: EntityExtractionSettings = Field(
        default_factory=EntityExtractionSettings
    )
    concept_linker: ConceptLinkerSettings = Field(default_factory=ConceptLinkerSettings)
    paths: PathsSettings
    model: ModelSettings
    tokenizer: TokenizerSettings
    global_: GlobalSettings = Field(alias="global")
    search_backend: SearchBackendSettings = Field(default_factory=SearchBackendSettings)
    patient_inputs: PatientInputSettings = Field(default_factory=PatientInputSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    embedder: EmbedderSettings
    cot: CotSettings
    LLM_reranker: LLMRerankerSettings
    search: SearchSettings
    constraints: ConstraintSettings = Field(default_factory=ConstraintSettings)
    query_expansion: QueryExpansionSettings = Field(
        default_factory=QueryExpansionSettings
    )
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    use_cot_reasoning: bool = True
    rag: RagSettings
    vllm: VllmSettings

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


def apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with environment variables for sensitive fields."""
    import os

    string_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_OUTPUT_DIR": ("paths", "output_dir"),
        "TRIALMATCHAI_TRIALS_JSON_FOLDER": ("paths", "trials_json_folder"),
        "TRIALMATCHAI_PATIENT_RAW_DIR": ("patient_inputs", "raw_dir"),
        "TRIALMATCHAI_PATIENT_PROFILE_DIR": ("patient_inputs", "profile_dir"),
        "TRIALMATCHAI_PATIENT_SUMMARY_DIR": ("patient_inputs", "summary_dir"),
        "TRIALMATCHAI_PATIENT_INPUT_FORMAT": ("patient_inputs", "default_format"),
        "TRIALMATCHAI_SEARCH_BACKEND": ("search_backend", "backend"),
        "TRIALMATCHAI_SEARCH_DB_PATH": ("search_backend", "db_path"),
        "TRIALMATCHAI_SEARCH_TRIALS_TABLE": ("search_backend", "trials_table"),
        "TRIALMATCHAI_SEARCH_CRITERIA_TABLE": ("search_backend", "criteria_table"),
        "TRIALMATCHAI_SEARCH_MODE": ("search", "mode"),
        "TRIALMATCHAI_EMBEDDER_MODEL_NAME": ("embedder", "model_name"),
        "TRIALMATCHAI_EMBEDDER_REVISION": ("embedder", "revision"),
        "TRIALMATCHAI_MODEL_BASE_MODEL": ("model", "base_model"),
        "TRIALMATCHAI_MODEL_BASE_MODEL_REVISION": (
            "model",
            "base_model_revision",
        ),
        "TRIALMATCHAI_MODEL_COT_ADAPTER_PATH": ("model", "cot_adapter_path"),
        "TRIALMATCHAI_QUERY_EXPANSION_MODEL": ("query_expansion", "model"),
        "TRIALMATCHAI_QUERY_EXPANSION_BACKEND": ("query_expansion", "backend"),
        "TRIALMATCHAI_QUERY_EXPANSION_ADAPTER": ("query_expansion", "adapter"),
        "TRIALMATCHAI_MODEL_RERANKER_MODEL_PATH": (
            "model",
            "reranker_model_path",
        ),
        "TRIALMATCHAI_MODEL_RERANKER_MODEL_REVISION": (
            "model",
            "reranker_model_revision",
        ),
        "TRIALMATCHAI_MODEL_RERANKER_ADAPTER_PATH": (
            "model",
            "reranker_adapter_path",
        ),
        "TRIALMATCHAI_ENTITY_BACKEND": ("entity_extraction", "backend"),
        "TRIALMATCHAI_ENTITY_MODEL_NAME": ("entity_extraction", "model_name"),
        "TRIALMATCHAI_ENTITY_MODEL_REVISION": (
            "entity_extraction",
            "model_revision",
        ),
        "TRIALMATCHAI_ENTITY_SCHEMA_PATH": ("entity_extraction", "schema_path"),
        "TRIALMATCHAI_ENTITY_DEVICE": ("entity_extraction", "device"),
        "TRIALMATCHAI_CONCEPT_DB_PATH": ("concept_linker", "db_path"),
        "TRIALMATCHAI_CONCEPT_TABLE": ("concept_linker", "table"),
        "TRIALMATCHAI_REGISTRY_SOURCE": ("registry", "source"),
        "TRIALMATCHAI_REGISTRY_API_BASE_URL": ("registry", "api_base_url"),
        "TRIALMATCHAI_REGISTRY_KEYWORDS_FILE": ("registry", "keywords_file"),
        "TRIALMATCHAI_REGISTRY_RAW_DIR": ("registry", "raw_dir"),
        "TRIALMATCHAI_REGISTRY_MANIFEST_PATH": ("registry", "manifest_path"),
        "TRIALMATCHAI_REGISTRY_REPORTS_DIR": ("registry", "reports_dir"),
    }
    for env_key, path in string_env_map.items():
        value = os.getenv(env_key)
        if value:
            _set_nested(raw, path, value)

    bool_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_EMBEDDER_USE_GPU": ("embedder", "use_gpu"),
        "TRIALMATCHAI_EMBEDDER_USE_FP16": ("embedder", "use_fp16"),
        "TRIALMATCHAI_EMBEDDER_TRUST_REMOTE_CODE": (
            "embedder",
            "trust_remote_code",
        ),
        "TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE": ("model", "trust_remote_code"),
        "TRIALMATCHAI_ENTITY_TRUST_REMOTE_CODE": (
            "entity_extraction",
            "trust_remote_code",
        ),
        "TRIALMATCHAI_CONCEPT_LINKER_ENABLED": ("concept_linker", "enabled"),
        "TRIALMATCHAI_PATIENT_STRICT_VALIDATION": (
            "patient_inputs",
            "strict_validation",
        ),
        "TRIALMATCHAI_PATIENT_COPY_RAW": ("patient_inputs", "copy_raw"),
        "TRIALMATCHAI_CONSTRAINTS_ENABLED": ("constraints", "enabled"),
        "TRIALMATCHAI_CONSTRAINTS_LLM_EXTRACTION_ENABLED": (
            "constraints",
            "llm_extraction_enabled",
        ),
        "TRIALMATCHAI_CONSTRAINTS_UNKNOWN_IS_NEUTRAL": (
            "constraints",
            "unknown_is_neutral",
        ),
        "TRIALMATCHAI_CONSTRAINTS_WRITE_REPORTS": ("constraints", "write_reports"),
        "TRIALMATCHAI_QUERY_EXPANSION_ENABLED": ("query_expansion", "enabled"),
        "TRIALMATCHAI_FIRST_LEVEL_ENABLED": ("search", "first_level", "enabled"),
        "TRIALMATCHAI_FIRST_LEVEL_LLM_EXPANSION_ENABLED": (
            "search",
            "first_level",
            "llm_expansion_enabled",
        ),
        "TRIALMATCHAI_FIRST_LEVEL_WRITE_REPORTS": (
            "search",
            "first_level",
            "write_reports",
        ),
    }
    for env_key, path in bool_env_map.items():
        value = os.getenv(env_key)
        if value:  # an empty/unset var must not be read as an explicit False
            _set_nested(raw, path, _parse_bool(value))

    int_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_SEARCH_CANDIDATE_LIMIT": (
            "search_backend",
            "candidate_limit",
        ),
        "TRIALMATCHAI_EMBEDDER_BATCH_SIZE": ("embedder", "batch_size"),
        "TRIALMATCHAI_ENTITY_BATCH_SIZE": ("entity_extraction", "batch_size"),
        "TRIALMATCHAI_CONCEPT_SEARCH_LIMIT": ("concept_linker", "search_limit"),
        "TRIALMATCHAI_REGISTRY_SINCE_DAYS": ("registry", "since_days"),
        "TRIALMATCHAI_REGISTRY_MAX_STUDIES": ("registry", "max_studies"),
        "TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL": (
            "search",
            "max_trials_first_level",
        ),
        "TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS": (
            "search",
            "first_level",
            "max_trials",
        ),
        "TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE": (
            "search",
            "first_level",
            "per_channel_size",
        ),
        "TRIALMATCHAI_FIRST_LEVEL_RRF_K": ("search", "first_level", "rrf_k"),
        "TRIALMATCHAI_FIRST_LEVEL_LLM_MAX_TERMS": (
            "search",
            "first_level",
            "llm_max_terms",
        ),
        "TRIALMATCHAI_SEARCH_MAX_TRIALS_SECOND_LEVEL": (
            "search",
            "max_trials_second_level",
        ),
        "TRIALMATCHAI_RAG_MAX_TRIALS": ("rag", "max_trials_rag"),
        "TRIALMATCHAI_VLLM_BATCH_SIZE": ("vllm", "batch_size"),
        "TRIALMATCHAI_VLLM_MAX_NEW_TOKENS": ("vllm", "max_new_tokens"),
    }
    for env_key, path in int_env_map.items():
        value = os.getenv(env_key)
        if value:
            try:
                _set_nested(raw, path, int(value))
            except ValueError:
                logger.warning("Ignoring malformed integer env override %s=%r", env_key, value)
    max_trials_first_level_env = os.getenv("TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL")
    if max_trials_first_level_env:
        try:
            _set_nested(
                raw,
                ("search", "first_level", "max_trials"),
                int(max_trials_first_level_env),
            )
        except ValueError:
            logger.warning(
                "Ignoring malformed TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL=%r",
                max_trials_first_level_env,
            )

    float_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_ENTITY_THRESHOLD": ("entity_extraction", "threshold"),
        "TRIALMATCHAI_LINK_ACCEPT": ("concept_linker", "accept_threshold"),
        "TRIALMATCHAI_LINK_REJECT": ("concept_linker", "reject_threshold"),
        "TRIALMATCHAI_REGISTRY_REQUEST_TIMEOUT": ("registry", "request_timeout"),
        "TRIALMATCHAI_REGISTRY_RATE_LIMIT_PER_SECOND": (
            "registry",
            "rate_limit_per_second",
        ),
        "TRIALMATCHAI_REGISTRY_FAILURE_THRESHOLD": (
            "registry",
            "failure_threshold",
        ),
        "TRIALMATCHAI_CONSTRAINTS_SCORE_WEIGHT": ("constraints", "score_weight"),
        "TRIALMATCHAI_FIRST_LEVEL_VECTOR_SCORE_THRESHOLD": (
            "search",
            "first_level",
            "vector_score_threshold",
        ),
    }
    for env_key, path in float_env_map.items():
        value = os.getenv(env_key)
        if value:
            try:
                _set_nested(raw, path, float(value))
            except ValueError:
                logger.warning("Ignoring malformed float env override %s=%r", env_key, value)

    return raw


def _set_nested(raw: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    keys = tuple(path)
    cursor: Dict[str, Any] = raw
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
