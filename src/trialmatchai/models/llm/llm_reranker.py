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
    ):
        from vllm import LLM, SamplingParams  # type: ignore

        try:
            from vllm.lora.request import LoRARequest  # type: ignore
        except ImportError:  # pragma: no cover - older vLLM
            LoRARequest = None  # type: ignore

        self.batch_size = batch_size
        enable_lora = bool(adapter_path) and LoRARequest is not None
        if adapter_path and not enable_lora:
            logger.warning(
                "Reranker adapter requested but LoRA is unavailable in this vLLM "
                "build; using the base model."
            )

        self.llm = LLM(
            model=str(model_path),
            revision=revision,
            trust_remote_code=trust_remote_code,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank if enable_lora else 16,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            # Single-token output: CUDA graphs buy nothing. enforce_eager skips graph capture
            # and a small max_num_seqs caps memory so this coexists with the CoT engine + embedder.
            enforce_eager=True,
            max_num_seqs=max(self.batch_size, 16),
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = (
            LoRARequest("reranker_adapter", 1, str(adapter_path)) if enable_lora else None
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
