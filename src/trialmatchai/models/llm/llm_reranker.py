import math
import re
import unicodedata
from typing import Any, Dict, List, Optional

from trialmatchai.utils.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging(__name__)


# Graded label space for "graded" scoring mode. The score is the Expected Relevance Value
# sum(p_k * y_k) over these label tokens, normalized to [0, 1].
#
# Why this exists. The default "binary" mode scores P(Yes) over a Yes/No pair, and three
# independent lines of evidence say that specific formulation is the weak link when scores are
# AGGREGATED rather than merely ordered -- which is exactly what aggregate_to_trials does:
#   * Setwise (SIGIR 2024): pointwise Yes/No reaches 0.386 BEIR nDCG@10, BELOW BM25 at 0.436.
#   * "Don't Overthink Passage Reranking" (2025): binary Yes/No collapses the partial-relevance
#     band -- ~0% of scores land in 0.1-0.9 versus 11.4% for a well-behaved reranker.
#   * ERank and TFRank (2025) were both built specifically because binary yes/no logprob
#     "lacks the necessary scoring discrimination", and both emit graded scores instead.
# "Beyond Yes and No" (NAACL 2024) measured the fix: replacing Yes/No with a 3-level scale and
# taking sum(p_k * y_k) gained +2.2 BEIR nDCG@10 on average and +7.1 on SciFact, at identical
# cost -- one forward pass, zero generated tokens. Gains flatten past ~7 levels.
#
# This matches an independently measured defect in this pipeline: shortlist_selection_delta is
# negative on every completed run, i.e. the second level currently selects WORSE than a plain
# first-level cut at the same depth.
GRADED_LABELS: tuple[str, ...] = ("Not", "Somewhat", "Highly")


class LLMReranker:
    """vLLM-backed pointwise reranker over (patient, criterion) pairs, optionally LoRA-adapted.

    Two scoring modes, selected by ``LLM_reranker.scoring``:

    ``binary`` (default)
        P(Yes) over a constrained Yes/No next token. Preserves the historical contract exactly.

    ``graded``
        Expected Relevance Value over ``GRADED_LABELS``: ``sum(p_k * y_k)`` normalized to
        [0, 1]. Same single forward pass and still zero generated tokens, but it yields a
        continuous magnitude instead of a saturated binary -- which is what the downstream
        aggregation and its 0.5 cut actually need.
    """

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
        scoring: str = "binary",
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

        self.scoring = str(scoring or "binary").lower()
        if self.scoring not in ("binary", "graded"):
            logger.warning(
                "Unknown LLM_reranker.scoring %r; falling back to 'binary'.", scoring
            )
            self.scoring = "binary"

        self.applicable_token_id, self.not_applicable_token_id = self._yes_no_token_ids()
        if self.scoring == "graded":
            self.label_token_ids = [self._first_token_id(w) for w in GRADED_LABELS]
            allowed = list(dict.fromkeys(self.label_token_ids))
            if len(allowed) != len(GRADED_LABELS):
                # Distinct labels must map to distinct first tokens or the softmax is degenerate.
                logger.error(
                    "Graded labels %s collide under this tokenizer; using binary scoring.",
                    GRADED_LABELS,
                )
                self.scoring = "binary"
        if self.scoring == "binary":
            self.label_token_ids = [
                self.not_applicable_token_id,
                self.applicable_token_id,
            ]
            allowed = list(self.label_token_ids)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=allowed,
        )
        logger.info("Reranker scoring mode: %s", self.scoring)

    def _first_token_id(self, word: str) -> int:
        return self.tokenizer(word, add_special_tokens=False)["input_ids"][0]

    def _yes_no_token_ids(self) -> tuple[int, int]:
        return self._first_token_id("Yes"), self._first_token_id("No")

    # Kept verbatim: this is the historical prompt and the LoRA adapter was tuned against it.
    # Note what it actually asks -- whether the patient text contains ENOUGH INFORMATION TO
    # EVALUATE the criterion. That is answerability, not relevance and not eligibility. It
    # correlates with relevance (a matching trial's criteria are more discussable) but is a
    # different quantity, and the downstream 0.5 cut therefore discards trials whose criteria
    # the patient text simply does not mention, however eligible the patient may be.
    BINARY_SYSTEM_PROMPT = (
        "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
        "provides enough details to evaluate whether the patient satisfies or violates the clinical "
        "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
        "information to make this evaluation, or 'No' if it does not."
    )

    # Asks for RELEVANCE on a graded scale. Two changes from the above, deliberately bundled
    # because the graded label space is what makes the relevance question aggregable.
    GRADED_SYSTEM_PROMPT = (
        "You are a clinical assistant. Judge how relevant the clinical trial eligibility "
        "criterion (Statement B) is to the patient described in Statement A -- that is, how "
        "much this criterion bears on whether this particular patient could join the trial.\n"
        "Answer with exactly one word:\n"
        "'Highly' - the criterion concerns this patient's condition, treatment or "
        "characteristics, and clearly bears on their eligibility.\n"
        "'Somewhat' - the criterion is related to the patient's clinical picture but is "
        "peripheral or only partly applicable.\n"
        "'Not' - the criterion concerns a different disease, population or context and does "
        "not bear on this patient."
    )

    @classmethod
    def create_messages(
        cls, patient_text: str, trial_text: str, *, scoring: str = "binary"
    ) -> List[Dict]:
        system_prompt = (
            cls.GRADED_SYSTEM_PROMPT if scoring == "graded" else cls.BINARY_SYSTEM_PROMPT
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
            self.preprocess_text(patient_text),
            self.preprocess_text(trial_text),
            scoring=self.scoring,
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _label_ids(self) -> list[int]:
        """Label tokens in ascending-relevance order.

        Falls back to [No, Yes] when ``label_token_ids`` is absent so an instance built
        without the full __init__ (older callers, hand-made test doubles) still scores as
        plain binary rather than raising.
        """
        ids = getattr(self, "label_token_ids", None)
        if ids:
            return list(ids)
        return [self.not_applicable_token_id, self.applicable_token_id]

    def _label_probabilities(self, output: Any) -> list[float] | None:
        """Softmax over the label tokens only, in ascending-relevance order."""
        try:
            token_logprobs = output.outputs[0].logprobs[0]
        except (AttributeError, IndexError, TypeError):
            return None
        logprobs = []
        for token_id in self._label_ids():
            entry = token_logprobs.get(token_id)
            logprobs.append(entry.logprob if entry is not None else float("-inf"))
        highest = max(logprobs)
        if highest == float("-inf"):
            return None
        exps = [math.exp(lp - highest) for lp in logprobs]
        total = sum(exps)
        if total <= 0:
            return None
        return [e / total for e in exps]

    def _yes_probability(self, output: Any) -> float:
        """Score in [0, 1].

        binary: P(Yes). graded: Expected Relevance Value sum(p_k * y_k) with y_k = k, divided
        by (K-1) so both modes share the [0, 1] range that aggregate_to_trials and its 0.5
        threshold assume. Renormalizing over the label tokens (rather than the full vocabulary)
        keeps the score comparable across prompts of different lengths.
        """
        probabilities = self._label_probabilities(output)
        if probabilities is None:
            return 0.0
        if getattr(self, "scoring", "binary") == "binary":
            # label_token_ids is [No, Yes]; P(Yes) reproduces the historical value exactly.
            return probabilities[1]
        k = len(probabilities) - 1
        if k <= 0:
            return 0.0
        return sum(p * i for i, p in enumerate(probabilities)) / k

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
