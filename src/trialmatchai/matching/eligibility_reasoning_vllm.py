from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class BatchTrialProcessorVLLM(BaseTrialProcessor):
    def __init__(
        self,
        llm: Any,
        tokenizer=None,
        batch_size: int = 16,
        use_cot: bool = True,
        max_new_tokens: int = 5000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 1234,
        length_bucket: bool = True,
        lora_request: Optional[Any] = None,
    ):
        """
        vLLM-backed trial processor for CoT eligibility evaluation.
        - Keeps long outputs (no custom stop).
        - Uses chat templates if tokenizer supports them.
        - Supports optional LoRA adapter via vLLM's LoRARequest.
        """
        self.llm = llm
        self.tokenizer = tokenizer or getattr(self.llm, "get_tokenizer", lambda: None)()
        self.batch_size = batch_size
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.length_bucket = length_bucket

        # Validate LoRA request during initialization
        self.lora_request = self._init_validate_lora_request(lora_request)

        from vllm import SamplingParams  # type: ignore

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            detokenize=True,
        )

    def _progress_desc(self) -> str:
        return "vLLM Processing Trials"

    def _init_validate_lora_request(self, lora_request):
        """Validate LoRA request during initialization."""
        if lora_request is None:
            return None

        try:
            # Check if LoRARequest has the expected attributes and types
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

            # Log batch info for debugging
            logger.debug(f"Processing batch with {len(prompts)} prompts")

            # Validate and safely pass LoRARequest
            safe_lora_request = self._validate_lora_request()

            # Generate with proper error handling for LoRA issues
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
                    # Retry without LoRA
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

                    # Safely extract input token count
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

                    # Safely extract output token count
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

            # Save outputs
            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

            # Safely calculate totals
            try:
                # Ensure all values are integers before summing
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
                # Create empty output files so processing can continue
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
            # Check if LoRARequest has the expected attributes
            if hasattr(self.lora_request, "lora_int_id"):
                lora_int_id = getattr(self.lora_request, "lora_int_id")

                # Fix string lora_int_id by converting to int
                if isinstance(lora_int_id, str):
                    try:
                        fixed_id = int(lora_int_id)
                        logger.warning(
                            f"Converting lora_int_id from string '{lora_int_id}' to int {fixed_id}"
                        )
                        # Try to set the corrected value
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
        # Default fallback based on character count
        fallback_length = max(1, len(prompt) // 4)

        if not self.length_bucket:
            return fallback_length

        try:
            # Try to use vLLM's built-in tokenizer first
            if hasattr(self.llm, "get_tokenizer"):
                vllm_tokenizer = self.llm.get_tokenizer()
                if vllm_tokenizer is not None:
                    try:
                        # vLLM tokenizers often have an encode method
                        if hasattr(vllm_tokenizer, "encode"):
                            token_ids = vllm_tokenizer.encode(prompt)
                            if isinstance(token_ids, (list, tuple)) or hasattr(
                                token_ids, "__len__"
                            ):
                                return int(len(token_ids))
                        elif hasattr(vllm_tokenizer, "__call__"):
                            # Fallback to callable tokenizer
                            result = vllm_tokenizer(prompt, add_special_tokens=False)
                            return self._extract_token_length(result, nct_id)
                    except Exception as e:
                        logger.warning(f"vLLM tokenizer failed for {nct_id}: {e}")

            # Fallback to the provided tokenizer
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

        # Try different extraction methods
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
