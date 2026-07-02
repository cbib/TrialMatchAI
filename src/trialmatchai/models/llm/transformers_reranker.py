from __future__ import annotations

import math
from typing import Any, List, Optional

from tqdm import tqdm

from trialmatchai.models.llm.llm_reranker import LLMReranker
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class TransformersReranker:
    """CPU-capable next-token Yes/No reranker for smoke tests and small runs."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Transformers reranker requires the llm extra "
                "(`uv sync --extra llm`)."
            ) from exc

        self.torch = torch
        self.device = torch.device(device if device != "auto" else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        # Left-pad so logits[:, -1, :] reads each prompt's real final token, not a pad token.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()
        self.yes_token_id = self._first_token_id(" Yes", "Yes")
        self.no_token_id = self._first_token_id(" No", "No")
        logger.info("Loaded Transformers reranker %s on %s.", model_path, self.device)

    def _first_token_id(self, preferred: str, fallback: str) -> int:
        token_ids = self.tokenizer(preferred, add_special_tokens=False)["input_ids"]
        if not token_ids:
            token_ids = self.tokenizer(fallback, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer could not encode {fallback!r}.")
        return int(token_ids[0])

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[dict[str, Any]]:
        results: List[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(patient_trial_pairs), self.batch_size),
            desc="CPU reranking batches",
        ):
            batch = patient_trial_pairs[start : start + self.batch_size]
            prompts = [self._build_prompt(patient, trial) for patient, trial in batch]
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**encoded)
                next_logits = outputs.logits[:, -1, :]
            for row in next_logits:
                yes_logit = float(row[self.yes_token_id])
                no_logit = float(row[self.no_token_id])
                highest = max(yes_logit, no_logit)
                yes = math.exp(yes_logit - highest)
                no = math.exp(no_logit - highest)
                prob = yes / (yes + no)
                results.append(
                    {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
                )
        return results

    def _build_prompt(self, patient_text: str, trial_text: str) -> str:
        messages = LLMReranker.create_messages(patient_text, trial_text)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return (
            f"{messages[0]['content']}\n\n"
            f"{messages[2]['content']}\nAnswer Yes or No:"
        )
