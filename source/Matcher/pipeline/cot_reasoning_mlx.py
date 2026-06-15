import json
import os
import time
from typing import Dict, List

from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file
from Matcher.utils.logging_config import setup_logging

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

logger = setup_logging(__name__)


class BatchTrialProcessorMLX:
    """
    CoT reasoning processor backed by mlx-lm.

    Runs on Apple Silicon only.  Expects (model, tokenizer) from
    Matcher.models.llm.mlx_loader.load_mlx_model().
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 1024,
        use_cot: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.use_cot = use_cot

    # ---------------------- I/O helpers ----------------------

    def _load_trial_data(self, nct_id: str, json_folder: str) -> str:
        try:
            trial_data = read_json_file(f"{json_folder}/{nct_id}.json")
            return trial_data.get("eligibility_criteria", "")
        except Exception as e:
            logger.error("Error loading %s: %s", nct_id, e)
            return ""

    # ---------------------- Prompting ----------------------

    def _format_prompt(self, criteria_text: str, patient_text: str) -> str:
        criteria_block = (
            f"Eligibility Criteria:\n{criteria_text}"
            if criteria_text
            else "No eligibility criteria provided."
        )

        if self.use_cot:
            system_msg = (
                "You are a medical expert with advanced knowledge in clinical reasoning, "
                "diagnostics, and treatment planning. Answer the following question. "
                "Before answering, create a concise chain of thoughts reasoning to ensure "
                "a logical and accurate response.\n"
            )
            user_msg = (
                "Assess the given patient's eligibility for a clinical trial by evaluating "
                "each and every criterion individually.\n\n"
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
                "### RESPONSE FORMAT (STRICTLY FOLLOW)\n"
                "{\n"
                '  "Inclusion_Criteria_Evaluation": [\n'
                '    {"Criterion": "...", "Classification": "Met | Not Met | Unclear | Irrelevant", "Justification": "..."}\n'
                "  ],\n"
                '  "Exclusion_Criteria_Evaluation": [\n'
                '    {"Criterion": "...", "Classification": "Violated | Not Violated | Unclear | Irrelevant", "Justification": "..."}\n'
                "  ],\n"
                '  "Recap": "...",\n'
                '  "Final Decision": "Eligible | Likely Eligible (leaning toward inclusion) | Likely Ineligible (leaning toward exclusion) | Ineligible"\n'
                "}\n\n"
                f"---Start of Clinical Trial Criteria---\n{criteria_block}\n---End of Clinical Trial Criteria---\n\n"
                f"---Start of Patient Description---\n{patient_text}\n"
                "Written informed consent has been obtained from the patient or their legal representative.\n"
                "---End of Patient Description---\n"
                "## IMPORTANT REMINDER:\n"
                "NEVER make assumptions, inferences, or extrapolations beyond the explicitly stated patient information."
            )
        else:
            system_msg = (
                "You are a clinical assistant tasked with assessing the eligibility of a patient "
                "for a clinical trial. Output only a JSON object.\n"
            )
            user_msg = (
                "For each criterion, classify:\n"
                '- Inclusion: "Met" | "Not Met" | "Unclear" | "Irrelevant"\n'
                '- Exclusion: "Violated" | "Not Violated" | "Unclear" | "Irrelevant"\n\n'
                "Output:\n"
                "{\n"
                '  "Inclusion_Criteria_Evaluation": [{"Criterion": "...", "Classification": "...", "Justification": "..."}],\n'
                '  "Exclusion_Criteria_Evaluation": [{"Criterion": "...", "Classification": "...", "Justification": "..."}],\n'
                '  "Final Decision": "Eligible | Likely Eligible | Likely Ineligible | Ineligible"\n'
                "}\n\n"
                f"---Start of Clinical Trial Criteria---\n{criteria_block}\n---End of Clinical Trial Criteria---\n\n"
                f"---Start of Patient Description---\n{patient_text}\n---End of Patient Description---\n"
            )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system_msg}\n\n{user_msg}\n\nAnswer: "
        # Prefill the assistant turn with '{' to force JSON output from the base model
        return prompt + "{"

    # ---------------------- Persistence ----------------------

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            os.makedirs(output_folder, exist_ok=True)
            write_text_file([response], f"{output_folder}/{nct_id}.txt")
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    json_data = json.loads(response[start : end + 1])
                    write_json_file(json_data, f"{output_folder}/{nct_id}.json")
                    logger.info("Processed %s successfully", nct_id)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON for %s: %s", nct_id, e)
            else:
                logger.error("No JSON boundaries found in response for %s", nct_id)
        except Exception as e:
            logger.error("Failed to save %s: %s", nct_id, e)

    # ---------------------- Public API ----------------------

    def process_trials(
        self,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_profile: List[str],
    ):
        if not _MLX_AVAILABLE:
            raise ImportError("mlx-lm not installed. Run: pip install mlx-lm")

        patient_text = " ".join(
            str(line).strip() for line in patient_profile if str(line).strip()
        )

        items: List[Dict] = []
        for nct_id in nct_ids:
            if os.path.exists(f"{output_folder}/{nct_id}.json"):
                logger.info("Skipping existing: %s", nct_id)
                continue
            criteria_text = self._load_trial_data(nct_id, json_folder)
            items.append({"nct_id": nct_id, "criteria": criteria_text})

        if not items:
            logger.info("No work to do.")
            return

        logger.info("MLX CoT: %d trial(s) to evaluate", len(items))

        for idx, item in enumerate(items, 1):
            nct_id = item["nct_id"]
            logger.info("  [%d/%d] %s", idx, len(items), nct_id)
            prompt = self._format_prompt(item["criteria"], patient_text)
            t0 = time.time()
            try:
                response = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    verbose=False,
                    sampler=mlx_make_sampler(temp=0.0),
                )
                # Re-attach the prefilled '{' that was part of the prompt
                response = "{" + response
                elapsed = time.time() - t0
                tok_count = len(response.split())
                logger.info(
                    "    → done in %.1fs (~%.1f tok/s)",
                    elapsed,
                    tok_count / elapsed if elapsed > 0 else 0,
                )
                self._save_outputs(nct_id, response, output_folder)
            except Exception:
                logger.exception("MLX generation failed for %s", nct_id)
