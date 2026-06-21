from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Tuple

from pydantic import BaseModel, Field, field_validator


class EntityExtractionSettings(BaseModel):
    backend: Literal["gliner2", "gliner", "regex", "disabled"] = "gliner2"
    model_name: str = "fastino/gliner2-base"
    fallback_model_name: str = "gliner-community/gliner_large-v2.5"
    model_revision: str | None = None
    schema_path: str = "source/Matcher/entity_schemas/trialmatchai.yaml"
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    batch_size: int = Field(8, ge=1)
    device: str = "auto"
    trust_remote_code: bool = False


class ConceptLinkerSettings(BaseModel):
    enabled: bool = True
    db_path: str = "data/concepts"
    table: str = "concepts"
    accept_threshold: float = Field(0.8, ge=0.0, le=1.0)
    reject_threshold: float = Field(0.3, ge=0.0, le=1.0)
    search_limit: int = Field(10, ge=1)

    @field_validator("reject_threshold")
    @classmethod
    def validate_reject_threshold(cls, value: float, info):
        accept = info.data.get("accept_threshold")
        if accept is not None and value > accept:
            raise ValueError("concept_linker.reject_threshold must be <= accept_threshold")
        return value


class PathsSettings(BaseModel):
    patients_dir: str
    output_dir: str
    trials_json_folder: str
    docker_certs: str


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


class ElasticsearchSettings(BaseModel):
    host: str
    username: str
    password: str
    request_timeout: int = 600
    retry_on_timeout: bool = True
    index_trials: str
    index_trials_eligibility: str
    auto_start: bool = False
    start_script: str = "elasticsearch/apptainer-run-es.sh"
    start_timeout: int = Field(600, ge=1)


class EmbedderSettings(BaseModel):
    model_name: str = "BAAI/bge-m3"
    revision: str | None = None
    trust_remote_code: bool = False
    pooling: str = "mean"
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = True
    use_fp16: bool = False
    normalize: bool = True

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, value: str) -> str:
        if value not in {"mean", "cls"}:
            raise ValueError("embedder.pooling must be 'mean' or 'cls'")
        return value


class SearchSettings(BaseModel):
    vector_score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_trials_first_level: int = Field(300, ge=1)
    max_trials_second_level: int = Field(100, ge=1)


class RagSettings(BaseModel):
    batch_size: int = Field(4, ge=1)
    max_trials_rag: int = Field(20, ge=1)


class VllmSettings(BaseModel):
    batch_size: int = Field(100, ge=1)
    max_new_tokens: int = Field(5000, ge=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    seed: int = 1234
    length_bucket: bool = True
    gpu_memory_utilization: float = Field(0.5, gt=0.0, le=1.0)
    max_model_len: int = Field(8192, ge=256)
    tensor_parallel_size: int = Field(1, ge=1)


class CotSettings(BaseModel):
    batch_size: int = Field(10, ge=1)


class LLMRerankerSettings(BaseModel):
    batch_size: int = Field(20, ge=1)


class TrialMatchSettings(BaseModel):
    entity_extraction: EntityExtractionSettings = Field(
        default_factory=EntityExtractionSettings
    )
    concept_linker: ConceptLinkerSettings = Field(default_factory=ConceptLinkerSettings)
    paths: PathsSettings
    model: ModelSettings
    tokenizer: TokenizerSettings
    global_: GlobalSettings = Field(alias="global")
    elasticsearch: ElasticsearchSettings
    embedder: EmbedderSettings
    cot: CotSettings
    LLM_reranker: LLMRerankerSettings
    search: SearchSettings
    use_cot_reasoning: bool = True
    cot_backend: str = "vllm"
    rag: RagSettings
    vllm: VllmSettings

    @field_validator("cot_backend")
    @classmethod
    def validate_cot_backend(cls, value: str) -> str:
        if value not in {"default", "vllm"}:
            raise ValueError("cot_backend must be 'default' or 'vllm'")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


def apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with environment variables for sensitive fields."""
    import os

    string_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_ES_HOST": ("elasticsearch", "host"),
        "TRIALMATCHAI_ES_USERNAME": ("elasticsearch", "username"),
        "TRIALMATCHAI_ES_PASSWORD": ("elasticsearch", "password"),
        "TRIALMATCHAI_ES_CA_CERTS": ("paths", "docker_certs"),
        "TRIALMATCHAI_PATIENTS_DIR": ("paths", "patients_dir"),
        "TRIALMATCHAI_OUTPUT_DIR": ("paths", "output_dir"),
        "TRIALMATCHAI_TRIALS_JSON_FOLDER": ("paths", "trials_json_folder"),
        "TRIALMATCHAI_INDEX_TRIALS": ("elasticsearch", "index_trials"),
        "TRIALMATCHAI_INDEX_TRIALS_ELIGIBILITY": (
            "elasticsearch",
            "index_trials_eligibility",
        ),
        "TRIALMATCHAI_EMBEDDER_MODEL_NAME": ("embedder", "model_name"),
        "TRIALMATCHAI_EMBEDDER_REVISION": ("embedder", "revision"),
        "TRIALMATCHAI_MODEL_BASE_MODEL": ("model", "base_model"),
        "TRIALMATCHAI_MODEL_BASE_MODEL_REVISION": (
            "model",
            "base_model_revision",
        ),
        "TRIALMATCHAI_MODEL_COT_ADAPTER_PATH": ("model", "cot_adapter_path"),
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
        "TRIALMATCHAI_COT_BACKEND": ("cot_backend",),
        "TRIALMATCHAI_ES_START_SCRIPT": ("elasticsearch", "start_script"),
        "TRIALMATCHAI_ENTITY_BACKEND": ("entity_extraction", "backend"),
        "TRIALMATCHAI_ENTITY_MODEL_NAME": ("entity_extraction", "model_name"),
        "TRIALMATCHAI_ENTITY_FALLBACK_MODEL_NAME": (
            "entity_extraction",
            "fallback_model_name",
        ),
        "TRIALMATCHAI_ENTITY_MODEL_REVISION": (
            "entity_extraction",
            "model_revision",
        ),
        "TRIALMATCHAI_ENTITY_SCHEMA_PATH": ("entity_extraction", "schema_path"),
        "TRIALMATCHAI_ENTITY_DEVICE": ("entity_extraction", "device"),
        "TRIALMATCHAI_CONCEPT_DB_PATH": ("concept_linker", "db_path"),
        "TRIALMATCHAI_CONCEPT_TABLE": ("concept_linker", "table"),
    }
    for env_key, path in string_env_map.items():
        value = os.getenv(env_key)
        if value:
            _set_nested(raw, path, value)

    bool_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_ES_AUTO_START": ("elasticsearch", "auto_start"),
        "TRIALMATCHAI_ES_RETRY_ON_TIMEOUT": ("elasticsearch", "retry_on_timeout"),
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
    }
    for env_key, path in bool_env_map.items():
        value = os.getenv(env_key)
        if value is not None:
            _set_nested(raw, path, _parse_bool(value))

    int_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_ES_REQUEST_TIMEOUT": ("elasticsearch", "request_timeout"),
        "TRIALMATCHAI_ES_START_TIMEOUT": ("elasticsearch", "start_timeout"),
        "TRIALMATCHAI_EMBEDDER_BATCH_SIZE": ("embedder", "batch_size"),
        "TRIALMATCHAI_ENTITY_BATCH_SIZE": ("entity_extraction", "batch_size"),
        "TRIALMATCHAI_CONCEPT_SEARCH_LIMIT": ("concept_linker", "search_limit"),
        "TRIALMATCHAI_SEARCH_MAX_TRIALS_FIRST_LEVEL": (
            "search",
            "max_trials_first_level",
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
                pass

    float_env_map: dict[str, Tuple[str, ...]] = {
        "TRIALMATCHAI_ENTITY_THRESHOLD": ("entity_extraction", "threshold"),
        "TRIALMATCHAI_LINK_ACCEPT": ("concept_linker", "accept_threshold"),
        "TRIALMATCHAI_LINK_REJECT": ("concept_linker", "reject_threshold"),
    }
    for env_key, path in float_env_map.items():
        value = os.getenv(env_key)
        if value:
            try:
                _set_nested(raw, path, float(value))
            except ValueError:
                pass

    return raw


def _set_nested(raw: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    keys = tuple(path)
    cursor: Dict[str, Any] = raw
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
