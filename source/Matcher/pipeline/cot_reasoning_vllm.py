from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file
from Matcher.utils.json_utils import extract_json_object
from Matcher.utils.logging_config import setup_logging
from tqdm import tqdm

from vllm import LLM, SamplingParams

try:
    # Present in vLLM when LoRA is enabled
    from vllm.lora.request import LoRARequest  # type: ignore
except Exception:  # pragma: no cover
    LoRARequest = None  # type: ignore

logger = setup_logging(__name__)


class BatchTrialProcessorVLLM:
    def __init__(
        self,
        llm: LLM,
        tokenizer=None,
        batch_size: int = 16,
        use_cot: bool = True,
        max_new_tokens: int = 5000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 1234,
        length_bucket: bool = True,
        lora_request: Optional[LoRARequest] = None,  # type: ignore
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

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            detokenize=True,
        )

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

    # ---------------------- I/O helpers ----------------------

    def _load_trial_data(self, nct_id: str, json_folder: str) -> str:
        try:
            path = f"{json_folder}/{nct_id}.json"
            trial_data = read_json_file(path)
            return trial_data.get("eligibility_criteria", "")
        except Exception as e:
            logger.error(f"Error loading {nct_id}: {str(e)}")
            return ""

    # ---------------------- Prompting ----------------------

    def _format_prompt(self, criteria_text: str, patient_profile: str) -> str:
        criteria_text_formatted = (
            f"Eligibility Criteria:\n{criteria_text}"
            if criteria_text
            else "No eligibility criteria provided."
        )

        if self.use_cot:
            system_msg = (
                "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
                "Answer the following question. Before answering, create a concise chain of thoughts reasoning to ensure a logical and accurate response.\n"
            )
            chat = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.\n\n"
                        "### INCLUSION CRITERIA ASSESSMENT\n"
                        "For each inclusion criterion, classify it as one of:\n"
                        "- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.\n"
                        "- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.\n"
                        "- **Unclear:** Insufficient or missing patient data to verify.\n"
                        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                        "### EXCLUSION CRITERIA ASSESSMENT\n"
                        "For each exclusion criterion, classify it as one of:\n"
                        "- **Violated:** The patient's data explicitly and unequivocally violates the criterion.\n"
                        "- **Not Violated:** The patient's data confirms compliance with the criterion.\n"
                        "- **Unclear:** Insufficient or missing patient data to verify.\n"
                        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                        "### IMPORTANT INSTRUCTIONS\n"
                        "- Ensure all criteria are assessed one-by-one.\n"
                        "- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.**\n"
                        "- Justifications must be strictly based on direct evidence from the patient profile.\n"
                        "### RESPONSE FORMAT (STRICTLY FOLLOW)\n"
                        "{\n"
                        '  "Inclusion_Criteria_Evaluation": [\n'
                        '    {"Criterion": "Exact inclusion criterion text", "Classification": "Met | Not Met | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                        "  ],\n"
                        '  "Exclusion_Criteria_Evaluation": [\n'
                        '    {"Criterion": "Exact exclusion criterion text", "Classification": "Violated | Not Violated | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                        "  ],\n"
                        '  "Recap": "Concise summary of key qualifying/disqualifying factors",\n'
                        '  "Final Decision": "Eligible | Likely Eligible (leaning toward inclusion) | Likely Ineligible (leaning toward exclusion) | Ineligible"\n'
                        "}\n\n"
                        "### INPUT\n"
                        "---Start of Clinical Trial Criteria---\n"
                        f"{criteria_text_formatted}\n"
                        "---End of Clinical Trial Criteria---\n\n"
                        "----\n"
                        "---Start of Patient Description---\n"
                        f"{patient_profile}\n"
                        "---End of Patient Description---\n"
                        "## IMPORTANT REMINDER:\n"
                        "NEVER make assumptions, inferences, or extrapolations beyond the explicitly stated patient information."
                    ),
                },
            ]
        else:
            chat = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant tasked with assessing the eligibility of a patient for a clinical trial. "
                        "Output only a JSON object evaluating trial eligibility for the patient based only on the provided criteria and patient profile.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "For each criterion, classify:\n"
                        '- If Inclusion Criterion: "Met" | "Not Met" | "Unclear" | "Irrelevant"\n'
                        '- If Exclusion Criterion: "Violated" | "Not Violated" | "Unclear" | "Irrelevant"\n\n'
                        "Provide a justification for each classification based strictly on the provided data. "
                        "Output this JSON schema:\n"
                        "{\n"
                        '  "Inclusion_Criteria_Evaluation": [ {"Criterion": "...", "Classification": "...", "Justification": "..."} ],\n'
                        '  "Exclusion_Criteria_Evaluation": [ {"Criterion": "...", "Classification": "...", "Justification": "..."} ],\n'
                        '  "Final Decision": "Eligible | Likely Eligible | Likely Ineligible | Ineligible"\n'
                        "}\n\n"
                        "---Start of Clinical Trial Criteria---\n"
                        f"{criteria_text_formatted}\n"
                        "---End of Clinical Trial Criteria---\n\n"
                        "---Start of Patient Description---\n"
                        f"{patient_profile}\n"
                        "---End of Patient Description---\n"
                    ),
                },
            ]

        if self.tokenizer is not None and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

        system_part = f"{chat[0]['content']}\n\n"
        user_part = f"{chat[1]['content']}\n\n"
        return system_part + user_part + "Answer: "

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

    # ---------------------- Persistence ----------------------

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            os.makedirs(output_folder, exist_ok=True)
            txt_path = f"{output_folder}/{nct_id}.txt"
            write_text_file([response], txt_path)
            try:
                json_data = extract_json_object(response)
                write_json_file(json_data, f"{output_folder}/{nct_id}.json")
                logger.info(f"Processed {nct_id} successfully")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid JSON response for {nct_id}: {str(e)}")
                write_json_file(
                    {"error": "invalid_json_response", "raw_output": response},
                    f"{output_folder}/{nct_id}.json",
                )
        except Exception as e:
            logger.error(f"Failed to save {nct_id}: {str(e)}")

    # ---------------------- Public API ----------------------

    def process_trials(
        self,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_profile: List[str],
    ):
        patient_text = " ".join(
            str(line).strip() for line in patient_profile if str(line).strip()
        )

        items: List[Dict] = []
        for nct_id in nct_ids:
            output_path = f"{output_folder}/{nct_id}.json"
            if os.path.exists(output_path):
                logger.info(f"Skipping existing: {nct_id}")
                continue

            criteria_text = self._load_trial_data(nct_id, json_folder)
            prompt = self._format_prompt(criteria_text, patient_text)

            # Calculate token length with comprehensive type safety
            tok_len = self._safe_calculate_token_length(prompt, nct_id)

            items.append({"nct_id": nct_id, "prompt": prompt, "tok_len": tok_len})

        if not items:
            logger.info("No work to do.")
            return

        # Validate all tok_len values before sorting
        self._validate_token_lengths(items)

        # Sort by token length if length_bucket is enabled
        if self.length_bucket:
            try:
                items.sort(key=lambda x: x["tok_len"])
                logger.debug(f"Successfully sorted {len(items)} items by token length")
            except Exception as e:
                logger.error(f"Failed to sort items by token length: {str(e)}")
                # Log the problematic items for debugging
                for i, item in enumerate(items):
                    logger.error(
                        f"Item {i}: nct_id={item['nct_id']}, tok_len={item['tok_len']}, type={type(item['tok_len'])}"
                    )
                # Disable length bucketing and continue without sorting
                logger.warning("Disabling length bucketing due to sorting failure")
                pass

        for i in tqdm(
            range(0, len(items), self.batch_size), desc="vLLM Processing Trials"
        ):
            batch = items[i : i + self.batch_size]
            self._process_batch(batch, output_folder)

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

    def _validate_token_lengths(self, items: List[Dict]) -> None:
        """Validate that all token lengths are integers and fix any that aren't."""
        for i, item in enumerate(items):
            tok_len = item["tok_len"]
            if not isinstance(tok_len, int):
                logger.error(
                    f"Invalid tok_len type for {item['nct_id']}: {type(tok_len)}, value: {tok_len}"
                )
                # Force conversion to int
                try:
                    if (
                        isinstance(tok_len, (float, str))
                        and str(tok_len).replace(".", "").isdigit()
                    ):
                        item["tok_len"] = int(float(tok_len))
                        logger.warning(
                            f"Converted tok_len for {item['nct_id']} from {type(tok_len)} to int"
                        )
                    else:
                        # Use character-based fallback
                        item["tok_len"] = max(1, len(item["prompt"]) // 4)
                        logger.warning(f"Using fallback tok_len for {item['nct_id']}")
                except Exception as e:
                    logger.error(f"Failed to convert tok_len for {item['nct_id']}: {e}")
                    item["tok_len"] = max(1, len(item["prompt"]) // 4)
