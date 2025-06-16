import os
import logging
import json
from typing import List
import math
from multiprocessing import Process
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from peft.peft_model import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Enable some CUDA and cuDNN optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class BatchTrialProcessor:
    """
    A class that processes clinical trial eligibility for a batch of NCT IDs
    using a language model to generate chain-of-thought reasoning and structured JSON output.
    """

    def __init__(self, base_model: str, device: int, batch_size: int = 4):
        """
        :param base_model: A string with the Hugging Face repository ID or local model path.
        :param device: GPU device ID to load the model onto.
        :param batch_size: Number of prompts to process in a batch.
        """
        self.device = device
        self.batch_size = batch_size
        self.base_model = base_model
        self.model, self.tokenizer = self._init_model()
        self.max_seq_length = 6000

    def _init_model(self):
        """
        Initializes the model and tokenizer using the provided base model.
        """
        try:
            # Bits and Bytes configuration for 4-bit quantization
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=f"cuda:{self.device}",
                attn_implementation="flash_attention_2",
                quantization_config=quant_config,
            )

            # Load PEFT-adapter (LoRA or other) on top of the base model
            model = PeftModel.from_pretrained(
                model, "finetuning/finetune_instruct_gemma2/finetuned_phi"
            )
            model.eval()

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize the model/tokenizer: {e}")
            raise

    def _load_trial_data(self, nct_id: str, json_folder: str) -> str:
        """
        Loads and returns the eligibility criteria text for a given trial (NCT ID)
        from a JSON file.
        """
        try:
            path = os.path.join(json_folder, f"{nct_id}.json")
            with open(path, "r", encoding="utf-8") as f:
                trial_data = json.load(f)
            criteria_text = trial_data.get("eligibility_criteria", "")
            return criteria_text
        except Exception as e:
            logger.error(f"Error loading {nct_id}: {str(e)}")
            return ""

    def _format_prompt(self, criteria_text: str, patient_profile: str) -> str:
        """
        Constructs a prompt string (including a system message and user instructions)
        for the clinical trial eligibility assessment using the eligibility criteria text.
        """
        criteria_text_formatted = (
            f"Eligibility Criteria:\n{criteria_text}"
            if criteria_text
            else "No eligibility criteria provided."
        )

        # System message: fosters chain-of-thought while maintaining clarity
        system_msg = (
            "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
            "Answer the following question. Before answering, create a step-by-step chain of thoughts to ensure a logical and accurate response.\n"
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

        # Use the tokenizer's chat template if available; otherwise, concatenate naively.
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            system_part = f"{chat[0]['content']}\n\n"
            user_part = f"{chat[1]['content']}\n\n"
            prompt = system_part + user_part + "Answer: "

        return prompt

    def _process_batch(self, batch: List[dict], output_folder: str):
        """
        Processes a batch of prompts through the model and saves output to text and JSON files.
        """
        try:
            # Tokenize the prompts in the batch
            inputs = self.tokenizer(
                [item["prompt"] for item in batch],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(f"cuda:{self.device}")

            # Generate responses
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=6000,
                    do_sample=False,
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode the model output
            decoded_responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Save each trial's output
            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            for item in batch:
                logger.error(f"Failed trial: {item['nct_id']}")

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        """
        Saves the raw text response to a .txt file and attempts to parse and save
        JSON data to a .json file.
        """
        try:
            txt_path = os.path.join(output_folder, f"{nct_id}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(response)

            # Attempt to parse JSON from the response (from the first '{' to the last '}')
            try:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                json_data = json.loads(json_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Invalid JSON response for {nct_id}: {str(e)}")
                return  # Skip saving if JSON is invalid

            json_path = os.path.join(output_folder, f"{nct_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            logger.info(f"Processed {nct_id} successfully")

        except Exception as e:
            logger.error(f"Failed to save {nct_id}: {str(e)}")

    def process_trials(
        self,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_profile: List[str],
    ):
        """
        Processes a list of trials sequentially on the GPU assigned to this instance.

        :param nct_ids: List of NCT ID strings to process.
        :param json_folder: Folder containing the JSON files for each trial.
        :param output_folder: Folder in which to save output .txt and .json files.
        :param patient_profile: A list of strings describing patient information.
        """
        patient_text = " ".join(
            str(line).strip() for line in patient_profile if str(line).strip()
        )

        current_batch = []
        for nct_id in tqdm(nct_ids, desc=f"GPU {self.device} Processing Trials"):
            output_path = os.path.join(output_folder, f"{nct_id}.json")
            if os.path.exists(output_path):
                logger.info(f"Skipping existing: {nct_id}")
                continue

            # Load trial eligibility criteria from JSON
            criteria_text = self._load_trial_data(nct_id, json_folder)

            # Format the prompt using the raw eligibility criteria text
            prompt = self._format_prompt(criteria_text, patient_text)

            current_batch.append({"nct_id": nct_id, "prompt": prompt})

            # Once the current batch is filled, process it
            if len(current_batch) >= self.batch_size:
                self._process_batch(current_batch, output_folder)
                current_batch = []

        # Process any leftover trials that didn't fill up the batch
        if current_batch:
            self._process_batch(current_batch, output_folder)

    @staticmethod
    def __parallel_worker(args):
        """
        A private static method used internally for parallel processing.
        Instantiates a new BatchTrialProcessor on the given GPU and processes its assigned trials.
        """
        (
            gpu_id,
            sub_nct_ids,
            base_model,
            batch_size,
            json_folder,
            output_folder,
            patient_profile,
        ) = args

        # Set the appropriate GPU for this process
        torch.cuda.set_device(gpu_id)

        logger.info(
            f"Parallel worker started on GPU {gpu_id} with {len(sub_nct_ids)} trials."
        )
        processor = BatchTrialProcessor(
            base_model=base_model, device=gpu_id, batch_size=batch_size
        )
        processor.process_trials(
            sub_nct_ids, json_folder, output_folder, patient_profile
        )

    @classmethod
    def parallel_process_trials(
        cls,
        base_model: str,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_profile: List[str],
        batch_size: int,
        gpus: List[int],
    ):
        """
        Distributes the processing of trials across multiple GPUs in parallel.

        :param base_model: Model name or path.
        :param nct_ids: List of NCT ID strings to process.
        :param json_folder: Folder containing the JSON files for each trial.
        :param output_folder: Folder in which to save output .txt and .json files.
        :param patient_profile: A list of strings describing patient information.
        :param batch_size: Number of prompts to process in a batch.
        :param gpus: List of GPU device IDs to use for parallel processing.
        """
        # Determine how many trials each GPU should process
        chunk_size = math.ceil(len(nct_ids) / len(gpus))
        tasks = []

        for i, gpu_id in enumerate(gpus):
            start = i * chunk_size
            sub_ids = nct_ids[start : start + chunk_size]
            if sub_ids:  # only add non-empty chunks
                tasks.append(
                    (
                        gpu_id,
                        sub_ids,
                        base_model,
                        batch_size,
                        json_folder,
                        output_folder,
                        patient_profile,
                    )
                )

        processes = []
        for task in tasks:
            p = Process(target=cls.__parallel_worker, args=(task,))
            processes.append(p)
            p.start()
            # Stagger process startup to reduce simultaneous GPU memory allocations
            time.sleep(5)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        logger.info("Parallel processing complete!")
