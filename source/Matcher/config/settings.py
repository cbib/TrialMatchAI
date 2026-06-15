from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class BioMedNerSettings(BaseModel):
    biomedner_port: int
    gner_port: int
    gene_norm_port: int
    disease_norm_port: int
    biomedner_home: str
    use_neural_normalizer: bool = True
    no_cuda: bool = False


class ServicesSettings(BaseModel):
    stop_script: str
    run_script: str


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
    mlx_merged_model_path: str = ""


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
    bio_med_ner: BioMedNerSettings
    services: ServicesSettings
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
        if value not in {"default", "vllm", "transformers", "mlx"}:
            raise ValueError("cot_backend must be 'default', 'vllm', 'transformers', or 'mlx'")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


def apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with environment variables for sensitive fields."""
    import os

    env_map = {
        "TRIALMATCHAI_ES_HOST": ("elasticsearch", "host"),
        "TRIALMATCHAI_ES_USERNAME": ("elasticsearch", "username"),
        "TRIALMATCHAI_ES_PASSWORD": ("elasticsearch", "password"),
        "TRIALMATCHAI_EMBEDDER_MODEL_NAME": ("embedder", "model_name"),
    }
    for env_key, path in env_map.items():
        value = os.getenv(env_key)
        if value:
            cursor: Dict[str, Any] = raw
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = value
    auto_start = os.getenv("TRIALMATCHAI_ES_AUTO_START")
    if auto_start is not None:
        cursor = raw.setdefault("elasticsearch", {})
        cursor["auto_start"] = auto_start.strip().lower() in {"1", "true", "yes"}

    start_script = os.getenv("TRIALMATCHAI_ES_START_SCRIPT")
    if start_script:
        cursor = raw.setdefault("elasticsearch", {})
        cursor["start_script"] = start_script

    start_timeout = os.getenv("TRIALMATCHAI_ES_START_TIMEOUT")
    if start_timeout:
        cursor = raw.setdefault("elasticsearch", {})
        try:
            cursor["start_timeout"] = int(start_timeout)
        except ValueError:
            pass
    return raw
