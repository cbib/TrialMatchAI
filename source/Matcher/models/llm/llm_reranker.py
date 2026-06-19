import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = setup_logging(__name__)


class LLMReranker:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: int = 0,
        torch_dtype=torch.float16,
        batch_size: int = 8,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        # Resolve device string
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            idx = int(device) if isinstance(device, int) else 0
            if idx < 0 or idx >= cuda_count:
                logger.warning(
                    f"LLMReranker: requested CUDA device {device} invalid; using 0 (num_gpus={cuda_count})."
                )
                idx = 0
            self.device_str = f"cuda:{idx}"
            # Ensure Accelerate/HF loaders use the selected GPU when device_map='auto'
            try:
                torch.cuda.set_device(idx)
            except Exception as e:
                logger.warning(f"Could not set CUDA device to {idx}: {e}")
        else:
            logger.warning("LLMReranker: CUDA not available; using CPU.")
            self.device_str = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        self._initialize_token_ids()
        self.model = self.load_model()
        self.model_lock = threading.Lock()

    def _initialize_token_ids(self):
        responses = ["Yes", "No"]
        token_ids = [
            self.tokenizer(response, add_special_tokens=False)["input_ids"]
            for response in responses
        ]
        self.applicable_token_id, self.not_applicable_token_id = [
            ids[0] for ids in token_ids
        ]

    def load_model(self):
        use_cuda = self.device_str.startswith("cuda")
        quant_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
            if use_cuda
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype if use_cuda else torch.float32,
            quantization_config=quant_config,
            device_map="auto" if use_cuda else None,
            attn_implementation="flash_attention_2" if use_cuda else None,
            trust_remote_code=self.trust_remote_code,
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return model

    def preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def create_messages(self, patient_text: str, trial_text: str) -> List[Dict]:
        system_prompt = (
            "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
            "provides enough details to evaluate whether the patient satisfies or violates the clinical "
            "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
            "information to make this evaluation, or 'No' if it does not."
        )
        return [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": " "},
            {
                "role": "user",
                "content": f"Statement A: {patient_text}\nStatement B: {trial_text}\n\n",
            },
        ]

    def process_batch(self, batch: List[tuple]) -> List[Dict]:
        batch_prompts = []
        for patient_text, trial_text in batch:
            messages = self.create_messages(
                self.preprocess_text(patient_text), self.preprocess_text(trial_text)
            )
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(prompt)
        inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        with self.model_lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probabilities = F.softmax(logits, dim=-1)
        applicable_probs = probabilities[:, self.applicable_token_id].tolist()
        return [
            {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
            for prob in applicable_probs
        ]

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[Dict]:
        batches = [
            patient_trial_pairs[i : i + self.batch_size]
            for i in range(0, len(patient_trial_pairs), self.batch_size)
        ]
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                results.extend(future.result())
        return results
