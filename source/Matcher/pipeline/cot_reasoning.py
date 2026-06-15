import json
import os
import time
from typing import Dict, List

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file
from Matcher.utils.logging_config import setup_logging
from tqdm import tqdm


class _TimeoutCriteria(StoppingCriteria):
    """Abort generation after max_seconds wall-clock time."""
    def __init__(self, max_seconds: float):
        self.deadline = time.time() + max_seconds

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return time.time() >= self.deadline

logger = setup_logging(__name__)


class BatchTrialProcessor:
    def __init__(
        self,
        model,
        tokenizer,
        device: int,
        batch_size: int = 4,
        use_cot: bool = True,
        max_new_tokens: int = 1024,
        generation_timeout: int = 300,  # seconds; 0 = no timeout
    ):
        """
        Optimized for throughput while preserving long outputs.

        Key improvements:
        - model.eval(), use_cache, TF32 hints, and SDPA/FlashAttention2 when available
        - length bucketing (sort by prompt token length) to reduce padding waste
        - telemetry for tokens/sec and stage timings
        """
        self.device = device
        # device may be an int (CUDA index), "mps", or "cpu"
        if isinstance(device, int):
            self.device_str = f"cuda:{device}"
        else:
            self.device_str = str(device)
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens
        self.generation_timeout = generation_timeout

        # ---- Inference-time performance knobs (safe if available) ----
        self.model.eval()
        try:
            # Allow TF32 on Ampere+ (gives a free speedup for matmuls with minimal accuracy loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Ensure caching is on for generation
        try:
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "use_cache"
            ):
                self.model.config.use_cache = True
        except Exception:
            pass

        # Pad token handling (avoids warnings for decoder-only models)
        try:
            if (
                self.tokenizer.pad_token_id is None
                and self.tokenizer.eos_token_id is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            pass

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
                        "Written informed consent has been obtained from the patient or their legal representative.\n"
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

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        system_part = f"{chat[0]['content']}\n\n"
        user_part = f"{chat[1]['content']}\n\n"
        return system_part + user_part + "Answer: "

    # ---------------------- Core batch path ----------------------

    def _process_batch(self, batch: List[Dict], output_folder: str):
        """
        Expects a list of dicts with keys: nct_id, prompt
        """
        try:
            ids = [item["nct_id"] for item in batch]
            logger.info("  → generating: %s", ", ".join(ids))
            t0 = time.time()
            # Tokenize once; pad to the longest in this batch only
            tokenized = self.tokenizer(
                [item["prompt"] for item in batch],
                padding=True,
                truncation=True,  # keep to model max length to avoid OOM
                return_tensors="pt",
            )
            input_len = tokenized["input_ids"].shape[1]
            logger.info("    input_len=%d tokens — starting generate (max_new=%d) ...",
                        input_len, self.max_new_tokens)
            tokenized = tokenized.to(self.device_str)
            t1 = time.time()

            # Autocast to model dtype if it's half/bfloat16 for extra speed
            model_dtype = next(self.model.parameters()).dtype
            # MPS doesn't support bfloat16 autocast; CUDA does float16/bfloat16
            use_autocast = (
                model_dtype in (torch.float16, torch.bfloat16)
                and self.device_str.startswith("cuda")
            )

            with torch.inference_mode():
                import contextlib
                ctx = (
                    torch.autocast(device_type="cuda", dtype=model_dtype)
                    if use_autocast
                    else contextlib.nullcontext()
                )
                stopping_criteria = None
                if self.generation_timeout > 0:
                    stopping_criteria = StoppingCriteriaList(
                        [_TimeoutCriteria(self.generation_timeout)]
                    )
                    logger.info("    timeout: %ds", self.generation_timeout)
                with ctx:
                    outputs = self.model.generate(
                        **tokenized,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        return_dict_in_generate=False,
                        stopping_criteria=stopping_criteria,
                    )
            t2 = time.time()

            # Decode only the generated tail
            gen_len = outputs.shape[1] - input_len
            decoded_responses = self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

            # Persist outputs
            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

            # Telemetry
            total_gen_tokens = gen_len * len(batch)
            gen_time = t2 - t1
            logger.info(
                f"[GPU {self.device}] batch={len(batch)} | in_len={input_len} | "
                f"out_len≈{gen_len} | tokenize+H2D={t1 - t0:.2f}s | "
                f"generate={gen_time:.2f}s | ~{(total_gen_tokens / gen_time) if gen_time > 0 else 0:.1f} tok/s"
            )
        except Exception:
            logger.exception("Batch processing failed for: %s", [item["nct_id"] for item in batch])

    # ---------------------- Persistence ----------------------

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            os.makedirs(output_folder, exist_ok=True)
            txt_path = f"{output_folder}/{nct_id}.txt"
            write_text_file([response], txt_path)
            try:
                # naive JSON slice (you may replace with a balanced-brace parser if needed)
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = response[start : end + 1]
                    json_data = json.loads(json_str)
                    write_json_file(json_data, f"{output_folder}/{nct_id}.json")
                    logger.info(f"Processed {nct_id} successfully")
                else:
                    logger.error(f"Invalid JSON boundaries for {nct_id}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response for {nct_id}: {str(e)}")
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
        """
        Length-buckets all prompts to reduce padding overhead, then processes in batches.
        Keeps existing outputs (idempotent resume).
        """
        patient_text = " ".join(
            str(line).strip() for line in patient_profile if str(line).strip()
        )

        # Build worklist, skipping already-processed trials
        items: List[Dict] = []
        for nct_id in nct_ids:
            output_path = f"{output_folder}/{nct_id}.json"
            if os.path.exists(output_path):
                logger.info(f"Skipping existing: {nct_id}")
                continue
            criteria_text = self._load_trial_data(nct_id, json_folder)
            prompt = self._format_prompt(criteria_text, patient_text)

            # Measure token length once for bucketing (no truncation here)
            try:
                tok = self.tokenizer(prompt, add_special_tokens=False)
                tok_len = len(tok["input_ids"])
            except Exception:
                # Fallback: rough char-based estimate if tokenizer hiccups
                tok_len = max(1, len(prompt) // 4)

            items.append({"nct_id": nct_id, "prompt": prompt, "tok_len": tok_len})

        if not items:
            logger.info("No work to do.")
            return

        # Sort by length (ascending) => minimal padding inside batches
        items.sort(key=lambda x: x["tok_len"])

        total = len(items)
        batches = [items[i : i + self.batch_size] for i in range(0, total, self.batch_size)]
        logger.info("CoT: %d trial(s) to evaluate, batch_size=%d → %d batch(es)",
                    total, self.batch_size, len(batches))
        for batch_idx, batch in enumerate(batches, 1):
            logger.info("  [%d/%d] %s", batch_idx, len(batches),
                        ", ".join(item["nct_id"] for item in batch))
            self._process_batch(batch, output_folder)
