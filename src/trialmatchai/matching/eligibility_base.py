"""Shared base for the CoT eligibility processors (HF and vLLM backends).

Holds prompting, trial I/O, output persistence, and worklist/bucketing. Backends
subclass this and implement ``_process_batch``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from trialmatchai.utils.file_utils import read_json_file, write_json_file, write_text_file
from trialmatchai.utils.json_utils import extract_json_object
from trialmatchai.utils.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging(__name__)


def _is_error_output(path: str) -> bool:
    """True if a per-trial output is a recorded failure or unparseable, so the resume
    worklist retries it instead of locking a transient failure into the ranking."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return True
    return isinstance(data, dict) and "error" in data


class BaseTrialProcessor:
    # Subclasses set these in __init__.
    tokenizer = None
    batch_size: int = 4
    use_cot: bool = True
    no_think: bool = False
    # Extra kwargs passed to apply_chat_template, e.g. {"thinking_mode": "on"} for models whose
    # template toggles reasoning by a named variable rather than enable_thinking (e.g. Baichuan-M2).
    chat_template_kwargs: dict | None = None

    # ---------------------- I/O helpers ----------------------

    def _load_trial_data(self, nct_id: str, json_folder) -> str:
        # json_folder may be one folder or an ordered fallback list so built,
        # bootstrapped, and updated trials all resolve.
        folders = [json_folder] if isinstance(json_folder, str) else list(json_folder)
        for folder in folders:
            try:
                trial_data = read_json_file(f"{folder}/{nct_id}.json")
            except Exception:
                continue
            criteria = trial_data.get("eligibility_criteria", "")
            if criteria:
                return criteria
        # Fail loud: no criteria found, else the model silently scores this trial 0.
        logger.error("No eligibility criteria found for %s in %s", nct_id, folders)
        return ""

    # ---------------------- Prompting ----------------------

    def _format_prompt(self, criteria_text: str, patient_profile: str) -> str:
        criteria_text_formatted = (
            f"Eligibility Criteria:\n{criteria_text}"
            if criteria_text
            else "No eligibility criteria provided."
        )

        no_think_prefix = "/no_think\n" if self.no_think else ""

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
                        no_think_prefix
                        + "Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.\n\n"
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
                        no_think_prefix
                        + "For each criterion, classify:\n"
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

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            # enable_thinking=False for reasoning models (e.g. Qwen3); templates that
            # don't declare the var raise TypeError — fall through to the plain prompt.
            template_kwargs: dict = dict(self.chat_template_kwargs or {})
            if self.no_think:
                template_kwargs["enable_thinking"] = False
            try:
                return self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True, **template_kwargs
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Tokenizer chat template unavailable; using plain prompt: %s",
                    exc,
                )
        # Fallback: simple concatenation
        system_part = f"{chat[0]['content']}\n\n"
        user_part = f"{chat[1]['content']}\n\n"
        return system_part + user_part + "Answer: "

    # ---------------------- Persistence ----------------------

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Drop the chain-of-thought preamble so only the JSON answer is parsed (the full
        response with thinking is still written to .txt).

        Reasoning models emit ``<think>…</think>``; the phi reasoning LoRA instead closes
        with a second ``<think>`` and emits the JSON after it. Treat either ``</think>`` or
        a second ``<think>`` as the close so the answer survives whether it precedes or
        follows the chain.
        """
        text = re.sub(r"<think>.*?(?:</think>|<think>)", "", text, flags=re.DOTALL)
        # A dangling <think> = cut off mid-thought after the answer; keep what precedes.
        cut = text.find("<think>")
        if cut != -1:
            text = text[:cut]
        return text.strip()

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            os.makedirs(output_folder, exist_ok=True)
            txt_path = f"{output_folder}/{nct_id}.txt"
            write_text_file([response], txt_path)
            try:
                json_data = extract_json_object(self._strip_thinking_tags(response))
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

    # ---------------------- Hooks for subclasses ----------------------

    def _token_length(self, prompt: str, nct_id: str = "") -> int:
        """Token count used for length bucketing; char-based fallback on error."""
        try:
            kwargs = {"add_special_tokens": False}
            max_input_tokens = getattr(self, "max_input_tokens", None)
            if isinstance(max_input_tokens, int) and max_input_tokens > 0:
                kwargs.update(truncation=True, max_length=max_input_tokens)
            return len(self.tokenizer(prompt, **kwargs)["input_ids"])
        except Exception:
            # Overestimate (~3 chars/token) so a tokenizer failure errs toward trimming.
            return max(1, len(prompt) // 3)

    def _progress_desc(self) -> str:
        return "Processing Trials"

    def _process_batch(self, batch: List[Dict], output_folder: str):  # pragma: no cover
        raise NotImplementedError

    # ---------------------- Public API ----------------------

    def process_trials(
        self,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_narrative: List[str],
    ):
        """Build a worklist (skipping already-processed trials), length-bucket to
        minimize padding, then process in batches."""
        patient_text = " ".join(
            str(line).strip() for line in patient_narrative if str(line).strip()
        )

        items: List[Dict] = []
        for nct_id in nct_ids:
            existing = f"{output_folder}/{nct_id}.json"
            # Skip only completed trials; recorded failures/unparseable files are retried.
            if os.path.exists(existing) and not _is_error_output(existing):
                logger.info(f"Skipping existing: {nct_id}")
                continue
            criteria_text = self._load_trial_data(nct_id, json_folder)
            prompt = self._format_prompt(criteria_text, patient_text)
            items.append(
                {
                    "nct_id": nct_id,
                    "prompt": prompt,
                    "tok_len": self._token_length(prompt, nct_id),
                }
            )

        if not items:
            logger.info("No work to do.")
            return

        if getattr(self, "length_bucket", True):
            items.sort(key=lambda x: x["tok_len"])

        for i in tqdm(
            range(0, len(items), self.batch_size), desc=self._progress_desc()
        ):
            self._process_batch(items[i : i + self.batch_size], output_folder)
