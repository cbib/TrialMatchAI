from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# JSON schema for grammar-constrained decoding (vLLM structured outputs). Forces every response to
# be a valid, complete eligibility object with the exact Met/Not Met/Unclear/Irrelevant (inclusion)
# and Violated/Not Violated/Unclear/Irrelevant (exclusion) vocabulary the trial ranker parses, so a
# verbose or reasoning model can never truncate mid-JSON or drift off-format.
_INCLUSION_LABELS = ["Met", "Not Met", "Unclear", "Irrelevant"]
_EXCLUSION_LABELS = ["Violated", "Not Violated", "Unclear", "Irrelevant"]


def _criteria_array(labels: list[str]) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Criterion": {"type": "string"},
                "Classification": {"type": "string", "enum": labels},
                "Justification": {"type": "string"},
            },
            "required": ["Criterion", "Classification", "Justification"],
        },
    }


ELIGIBILITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "Inclusion_Criteria_Evaluation": _criteria_array(_INCLUSION_LABELS),
        "Exclusion_Criteria_Evaluation": _criteria_array(_EXCLUSION_LABELS),
        "Final Decision": {
            "type": "string",
            "enum": ["Eligible", "Likely Eligible", "Likely Ineligible", "Ineligible"],
        },
    },
    "required": ["Inclusion_Criteria_Evaluation", "Exclusion_Criteria_Evaluation", "Final Decision"],
}


class BatchTrialProcessorVLLM(BaseTrialProcessor):
    def __init__(
        self,
        llm: Any,
        tokenizer=None,
        batch_size: int = 16,
        use_cot: bool = True,
        no_think: bool = False,
        max_new_tokens: int = 5000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 1234,
        length_bucket: bool = True,
        max_model_len: Optional[int] = None,
        lora_request: Optional[Any] = None,
        chat_template_kwargs: Optional[dict] = None,
        guided_json: bool = False,
    ):
        """vLLM-backed CoT eligibility processor with optional LoRA adapter."""
        self.llm = llm
        self.tokenizer = tokenizer or getattr(self.llm, "get_tokenizer", lambda: None)()
        self.batch_size = batch_size
        self.use_cot = use_cot
        self.no_think = no_think
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.length_bucket = length_bucket
        # Reserve output room so a long-criteria prompt can't push the response past the
        # context window (dropped as invalid JSON); unknown window (None) disables trimming.
        self.max_model_len = max_model_len
        if max_model_len and max_model_len > 0:
            reserved = min(self.max_new_tokens, max(1, max_model_len // 4))
            self._max_prompt_tokens: Optional[int] = max(1, max_model_len - reserved)
        else:
            self._max_prompt_tokens = None

        self.lora_request = self._init_validate_lora_request(lora_request)

        from vllm import SamplingParams  # type: ignore

        # Grammar-constrained decoding: guarantee a complete, schema-valid eligibility JSON so a
        # verbose/reasoning model can't truncate or drift off-format (xgrammar backend, auto).
        structured = None
        if guided_json:
            from vllm.sampling_params import StructuredOutputsParams  # type: ignore

            structured = StructuredOutputsParams(json=ELIGIBILITY_JSON_SCHEMA)
            logger.info("Eligibility decoding constrained to the JSON schema (vLLM structured outputs).")

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            detokenize=True,
            structured_outputs=structured,
        )

    def _progress_desc(self) -> str:
        return "vLLM Processing Trials"

    def _format_prompt(self, criteria_text: str, patient_profile: str) -> str:
        # Trim only the criteria text (long/variable) so the prompt leaves room to generate;
        # keeps the schema + patient profile intact.
        prompt = super()._format_prompt(criteria_text, patient_profile)
        budget = getattr(self, "_max_prompt_tokens", None)
        if not budget or not criteria_text or self.tokenizer is None:
            return prompt
        try:
            # Tokenize directly (not _token_length, which may short-circuit to a char
            # estimate) so the fit check and criterion_ids trim math share one token unit.
            n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if n <= budget:
                return prompt
            criterion_ids = self.tokenizer(criteria_text, add_special_tokens=False)["input_ids"]
            keep = len(criterion_ids) - (n - budget) - 32  # 32-token chat-template slack
            if keep <= 0:
                # Profile alone exhausts the budget; an empty criteria block is worse than
                # the untrimmed prompt, so keep the criteria and accept tail-truncation risk.
                return prompt
            trimmed = self.tokenizer.decode(criterion_ids[:keep])
            logger.warning(
                "Trimmed long eligibility criteria to fit the context window "
                "(prompt ~%s tok > budget %s); kept ~%s of %s criteria tokens.",
                n, budget, keep, len(criterion_ids),
            )
            return super()._format_prompt(trimmed, patient_profile)
        except Exception as exc:  # never let trimming break generation
            logger.warning("Prompt-fit trimming failed (%s); using the untrimmed prompt.", exc)
            return prompt

    def _init_validate_lora_request(self, lora_request):
        """Validate LoRA request during initialization."""
        if lora_request is None:
            return None

        try:
            if hasattr(lora_request, "lora_int_id"):
                lora_int_id = getattr(lora_request, "lora_int_id")

                if isinstance(lora_int_id, str):
                    try:
                        fixed_id = int(lora_int_id)
                        logger.warning(
                            f"Fixed lora_int_id during init: '{lora_int_id}' -> {fixed_id}"
                        )
                        setattr(lora_request, "lora_int_id", fixed_id)
                    except (ValueError, AttributeError) as e:
                        logger.error(f"Cannot fix lora_int_id during init: {e}")
                        logger.warning("Disabling LoRA due to invalid lora_int_id")
                        return None
                elif not isinstance(lora_int_id, int):
                    logger.error(
                        f"lora_int_id has invalid type during init: {type(lora_int_id)}"
                    )
                    logger.warning("Disabling LoRA due to invalid lora_int_id type")
                    return None

            logger.info("LoRA request validated successfully")
            return lora_request

        except Exception as e:
            logger.error(f"Error validating LoRARequest during init: {e}")
            logger.warning("Disabling LoRA due to validation error")
            return None

    # ---------------------- Core batch path (vLLM) ----------------------

    def _process_batch(self, batch: List[Dict], output_folder: str):
        try:
            prompts = [item["prompt"] for item in batch]
            t0 = time.time()

            logger.debug(f"Processing batch with {len(prompts)} prompts")

            safe_lora_request = self._validate_lora_request()

            try:
                results = self.llm.generate(
                    prompts,
                    self.sampling_params,
                    lora_request=safe_lora_request,
                )
            except TypeError as e:
                if "not supported between instances of 'str' and 'int'" in str(e):
                    logger.warning(f"LoRA configuration issue detected: {e}")
                    logger.warning("Retrying without LoRA request...")
                    results = self.llm.generate(
                        prompts,
                        self.sampling_params,
                        lora_request=None,
                    )
                else:
                    raise

            t1 = time.time()

            decoded_responses: List[str] = []
            in_tok_lens: List[int] = []
            out_tok_lens: List[int] = []

            for i, r in enumerate(results):
                try:
                    text = r.outputs[0].text if r.outputs else ""
                    decoded_responses.append(text)

                    try:
                        prompt_token_ids = getattr(r, "prompt_token_ids", []) or []
                        if isinstance(prompt_token_ids, (list, tuple)):
                            in_tok_count = len(prompt_token_ids)
                        elif hasattr(prompt_token_ids, "__len__"):
                            in_tok_count = len(prompt_token_ids)
                        else:
                            logger.warning(
                                f"Unexpected prompt_token_ids type for result {i}: {type(prompt_token_ids)}"
                            )
                            in_tok_count = 0
                        in_tok_lens.append(int(in_tok_count))
                    except Exception as e:
                        logger.warning(
                            f"Failed to get input token count for result {i}: {e}"
                        )
                        in_tok_lens.append(0)

                    try:
                        if r.outputs and len(r.outputs) > 0:
                            output_token_ids = (
                                getattr(r.outputs[0], "token_ids", []) or []
                            )
                            if isinstance(output_token_ids, (list, tuple)):
                                out_tok_count = len(output_token_ids)
                            elif hasattr(output_token_ids, "__len__"):
                                out_tok_count = len(output_token_ids)
                            else:
                                logger.warning(
                                    f"Unexpected output token_ids type for result {i}: {type(output_token_ids)}"
                                )
                                out_tok_count = 0
                            out_tok_lens.append(int(out_tok_count))
                        else:
                            out_tok_lens.append(0)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get output token count for result {i}: {e}"
                        )
                        out_tok_lens.append(0)

                except Exception as e:
                    logger.error(f"Failed to process result {i}: {e}")
                    decoded_responses.append("")
                    in_tok_lens.append(0)
                    out_tok_lens.append(0)

            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

            try:
                safe_in_tok_lens = [
                    int(x)
                    if isinstance(x, (int, float, str))
                    and str(x).replace(".", "").isdigit()
                    else 0
                    for x in in_tok_lens
                ]
                safe_out_tok_lens = [
                    int(x)
                    if isinstance(x, (int, float, str))
                    and str(x).replace(".", "").isdigit()
                    else 0
                    for x in out_tok_lens
                ]

                total_in = sum(safe_in_tok_lens)
                total_out = sum(safe_out_tok_lens)
                gen_time = max(1e-6, t1 - t0)

                logger.info(
                    f"[vLLM] batch={len(batch)} | in_tok≈{total_in} | out_tok≈{total_out} | "
                    f"elapsed={gen_time:.2f}s | ~{(total_out / gen_time):.1f} tok/s"
                )
            except Exception as e:
                logger.error(f"Failed to calculate token statistics: {e}")
                logger.info(f"[vLLM] batch={len(batch)} completed (stats unavailable)")

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            for item in batch:
                logger.error(f"Failed trial: {item['nct_id']}")
                # Write an error marker so the run continues.
                try:
                    self._save_outputs(
                        item["nct_id"], '{"error": "processing_failed"}', output_folder
                    )
                except Exception as save_e:
                    logger.error(
                        f"Failed to save error output for {item['nct_id']}: {save_e}"
                    )

    def _validate_lora_request(self):
        """Validate and fix LoRARequest to prevent vLLM type errors."""
        if self.lora_request is None:
            return None

        try:
            if hasattr(self.lora_request, "lora_int_id"):
                lora_int_id = getattr(self.lora_request, "lora_int_id")

                # Coerce a string lora_int_id to int (vLLM compares it against ints).
                if isinstance(lora_int_id, str):
                    try:
                        fixed_id = int(lora_int_id)
                        logger.warning(
                            f"Converting lora_int_id from string '{lora_int_id}' to int {fixed_id}"
                        )
                        setattr(self.lora_request, "lora_int_id", fixed_id)
                    except (ValueError, AttributeError) as e:
                        logger.error(f"Failed to fix lora_int_id: {e}")
                        logger.warning("Disabling LoRA due to invalid lora_int_id")
                        return None
                elif not isinstance(lora_int_id, int):
                    logger.error(f"lora_int_id has invalid type: {type(lora_int_id)}")
                    logger.warning("Disabling LoRA due to invalid lora_int_id type")
                    return None

            return self.lora_request

        except Exception as e:
            logger.error(f"Error validating LoRARequest: {e}")
            logger.warning("Disabling LoRA due to validation error")
            return None

    # ---------------------- Token-length bucketing (vLLM tokenizer aware) ----

    def _token_length(self, prompt: str, nct_id: str = "") -> int:
        return self._safe_calculate_token_length(prompt, nct_id)

    def _safe_calculate_token_length(self, prompt: str, nct_id: str) -> int:
        """Safely calculate token length using vLLM's tokenizer."""
        fallback_length = max(1, len(prompt) // 4)

        if not self.length_bucket:
            return fallback_length

        try:
            if hasattr(self.llm, "get_tokenizer"):
                vllm_tokenizer = self.llm.get_tokenizer()
                if vllm_tokenizer is not None:
                    try:
                        if hasattr(vllm_tokenizer, "encode"):
                            token_ids = vllm_tokenizer.encode(prompt)
                            if isinstance(token_ids, (list, tuple)) or hasattr(
                                token_ids, "__len__"
                            ):
                                return int(len(token_ids))
                        elif hasattr(vllm_tokenizer, "__call__"):
                            result = vllm_tokenizer(prompt, add_special_tokens=False)
                            return self._extract_token_length(result, nct_id)
                    except Exception as e:
                        logger.warning(f"vLLM tokenizer failed for {nct_id}: {e}")

            if self.tokenizer is not None:
                tokenized = self.tokenizer(prompt, add_special_tokens=False)
                return self._extract_token_length(tokenized, nct_id)

            return fallback_length

        except Exception as e:
            logger.warning(
                f"All tokenization methods failed for {nct_id}: {str(e)}, using character-based estimate"
            )
            return fallback_length

    def _extract_token_length(self, tokenized, nct_id: str) -> int:
        """Extract token length from various tokenizer output formats."""
        fallback_length = max(1, len(str(tokenized)) // 4)

        extraction_methods = [
            lambda x: len(x["input_ids"])
            if isinstance(x, dict) and "input_ids" in x
            else None,
            lambda x: len(x.input_ids) if hasattr(x, "input_ids") else None,
            lambda x: len(x) if isinstance(x, (list, tuple)) else None,
            lambda x: len(x)
            if hasattr(x, "__len__") and not isinstance(x, (str, dict))
            else None,
        ]

        for method in extraction_methods:
            try:
                result = method(tokenized)
                if (
                    result is not None
                    and isinstance(result, (int, float))
                    and result > 0
                ):
                    return int(result)
            except Exception:
                continue

        logger.warning(f"Could not extract token length for {nct_id}, using fallback")
        return fallback_length
