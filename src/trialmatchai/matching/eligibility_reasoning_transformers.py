from __future__ import annotations

from typing import Any, Dict, List, Optional

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class BatchTrialProcessorTransformers(BaseTrialProcessor):
    """CPU-capable Transformers eligibility processor for smoke/small runs."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 1,
        use_cot: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        length_bucket: bool = True,
        no_think: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Transformers eligibility processing requires the llm extra "
                "(`uv sync --extra llm`)."
            ) from exc

        self.torch = torch
        self.device = torch.device(device if device != "auto" else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.length_bucket = length_bucket
        self.no_think = no_think
        self.max_input_tokens = _max_input_tokens(
            self.tokenizer,
            self.model,
            self.max_new_tokens,
        )
        logger.info(
            "Loaded Transformers eligibility model %s on %s.",
            model_path,
            self.device,
        )

    def _progress_desc(self) -> str:
        return "CPU Transformers Processing Trials"

    def _process_batch(self, batch: List[Dict[str, Any]], output_folder: str):
        prompts = [item["prompt"] for item in batch]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generation_kwargs["do_sample"] = False

        with self.torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)

        prompt_width = encoded["input_ids"].shape[1]
        for item, sequence in zip(batch, output_ids):
            completion_ids = sequence[prompt_width:]
            response = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip()
            self._save_outputs(item["nct_id"], response, output_folder)


def _max_input_tokens(tokenizer: Any, model: Any, max_new_tokens: int) -> int:
    candidates: list[int] = []
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)

    context_window = min(candidates) if candidates else 2048
    return max(1, context_window - max(0, int(max_new_tokens)))
