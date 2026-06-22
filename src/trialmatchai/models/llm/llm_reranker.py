import re
import unicodedata
from typing import Any, Dict, List, Optional

from trialmatchai.models.llm._common import (
    build_4bit_quant_config,
    configure_decoder_tokenizer,
    load_llm_dependencies,
    resolve_cuda_device,
    select_attn_impl,
    select_compute_dtype,
)
from trialmatchai.utils.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging(__name__)


class LLMReranker:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: Any = 0,
        torch_dtype: Any | None = None,
        batch_size: int = 8,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self._deps = load_llm_dependencies()
        self._torch = self._deps.torch
        self._torch_functional = self._deps.torch_functional
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.batch_size = batch_size
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        # Validate/select the GPU once (handles int or "auto"); pins the device so
        # device_map below is consistent with where inputs are moved.
        self.device_str, self._cuda_index = resolve_cuda_device(
            self._torch, device, label="LLMReranker"
        )
        use_cuda = self._cuda_index is not None
        self.torch_dtype = torch_dtype or select_compute_dtype(self._torch, use_cuda)

        self.tokenizer = self._deps.auto_tokenizer.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        # Left padding is required: process_batch reads logits[:, -1, :], which
        # must be the last real token, not a right-pad position.
        configure_decoder_tokenizer(self.tokenizer)
        self._initialize_token_ids()
        self.model = self.load_model()

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
        use_cuda = self._cuda_index is not None
        quant_config = (
            build_4bit_quant_config(
                self._deps.bnb_config,
                load_in_4bit=True,
                double_quant=True,
                quant_type="nf4",
                compute_dtype=self.torch_dtype,
            )
            if use_cuda
            else None
        )
        model = self._deps.auto_model.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.torch_dtype if use_cuda else self._torch.float32,
            quantization_config=quant_config,
            # Pin to the selected GPU (not "auto"): inputs are moved to
            # self.device_str, so the model's first layer must live there too.
            device_map=self.device_str if use_cuda else None,
            attn_implementation=select_attn_impl(self._torch, self._cuda_index),
            trust_remote_code=self.trust_remote_code,
        )
        if self.adapter_path:
            model = self._deps.peft_model.from_pretrained(model, self.adapter_path)
        model.eval()
        return model

    def preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def create_messages(patient_text: str, trial_text: str) -> List[Dict]:
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
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probabilities = self._torch_functional.softmax(logits, dim=-1)
        applicable_probs = probabilities[:, self.applicable_token_id].tolist()
        return [
            {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
            for prob in applicable_probs
        ]

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[Dict]:
        # Inference on a single device is serial regardless; iterate batches
        # directly rather than behind a thread pool + lock that serialized anyway.
        batches = [
            patient_trial_pairs[i : i + self.batch_size]
            for i in range(0, len(patient_trial_pairs), self.batch_size)
        ]
        results: List[Dict] = []
        for batch in tqdm(batches, desc="Reranking batches"):
            results.extend(self.process_batch(batch))
        return results
