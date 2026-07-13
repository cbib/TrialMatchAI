import math
import re
import unicodedata
from typing import Any, Dict, List, Optional

from trialmatchai.utils.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging(__name__)


class LLMReranker:
    """vLLM-backed pointwise reranker: scores each (patient, criterion) pair by the constrained
    Yes/No next-token probability, optionally via a LoRA adapter."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: Any = 0,  # accepted for API compatibility; vLLM manages devices
        torch_dtype: Any | None = None,
        batch_size: int = 8,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 4096,
        max_lora_rank: int = 32,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        quantization: str = "",
        kv_cache_dtype: str | None = None,
    ):
        from vllm import SamplingParams  # type: ignore

        # Build via the SHARED vLLM loader (not an inline vllm.LLM) so the reranker inherits the
        # CoT engine's knobs (quantization, kv_cache_dtype, max_model_len cap, LoRARequest
        # fallback) and joins the one engine cache. Scoring is unchanged.
        from trialmatchai.models.llm.vllm_loader import load_vllm_engine

        self.batch_size = batch_size
        model_config = {
            "base_model": str(model_path),
            "cot_adapter_path": str(adapter_path) if adapter_path else None,
            "trust_remote_code": trust_remote_code,
            "base_model_revision": revision,
        }
        vllm_cfg = {
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "max_lora_rank": max_lora_rank,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
            # Single-token output: CUDA graphs buy nothing. enforce_eager skips graph capture and a
            # small max_num_seqs caps memory so this coexists with the CoT engine + embedder.
            "enforce_eager": True,
            "max_num_seqs": max(self.batch_size, 16),
            "adapter_name": "reranker_adapter",
        }
        self.llm, self.tokenizer, self.lora_request = load_vllm_engine(
            model_config=model_config, vllm_cfg=vllm_cfg
        )

        self.applicable_token_id, self.not_applicable_token_id = self._yes_no_token_ids()
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.applicable_token_id, self.not_applicable_token_id],
        )

    def _yes_no_token_ids(self) -> tuple[int, int]:
        yes = self.tokenizer("Yes", add_special_tokens=False)["input_ids"]
        no = self.tokenizer("No", add_special_tokens=False)["input_ids"]
        return yes[0], no[0]

    @staticmethod
    def create_messages(patient_text: str, trial_text: str) -> List[Dict]:
        system_prompt = (
            "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
            "provides enough details to evaluate whether the patient satisfies or violates the clinical "
            "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
            "information to make this evaluation, or 'No' if it does not."
        )
        return [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": " "},
            {
                "role": "user",
                "content": f"Statement A: {patient_text}\nStatement B: {trial_text}\n\n",
            },
        ]

    def preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_prompt(self, patient_text: str, trial_text: str) -> str:
        messages = self.create_messages(
            self.preprocess_text(patient_text), self.preprocess_text(trial_text)
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _yes_probability(self, output: Any) -> float:
        try:
            token_logprobs = output.outputs[0].logprobs[0]
        except (AttributeError, IndexError, TypeError):
            return 0.0
        yes = token_logprobs.get(self.applicable_token_id)
        no = token_logprobs.get(self.not_applicable_token_id)
        yes_lp = yes.logprob if yes is not None else float("-inf")
        no_lp = no.logprob if no is not None else float("-inf")
        highest = max(yes_lp, no_lp)
        if highest == float("-inf"):
            return 0.0
        ey = math.exp(yes_lp - highest)
        en = math.exp(no_lp - highest)
        return ey / (ey + en)

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[Dict]:
        results: List[Dict] = []
        for start in tqdm(
            range(0, len(patient_trial_pairs), self.batch_size),
            desc="Reranking batches",
        ):
            batch = patient_trial_pairs[start : start + self.batch_size]
            prompts = [self._build_prompt(p, t) for p, t in batch]
            outputs = self.llm.generate(
                prompts, self.sampling_params, lora_request=self.lora_request
            )
            for output in outputs:
                prob = self._yes_probability(output)
                results.append(
                    {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
                )
        return results
