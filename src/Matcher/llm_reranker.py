import torch
import torch.nn.functional as F
import unicodedata
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from peft.peft_model import PeftModel
import threading


# Configure torch settings for optimal performance.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class LLMReranker:
    """
    Classifies patient–trial-criterion pairs with answers: 'Yes' and 'No'
    using a single preloaded model shared across threads.
    """

    def __init__(
        self,
        model_path,
        adapter_path=None,
        device=0,
        torch_dtype=torch.float16,
        batch_size=8,
    ):
        """
        Initializes the LLMReranker class.

        Parameters:
            model_path (str): Path or identifier of the base model.
            adapter_path (str): Path to the PEFT adapter.
            device (int): CUDA device ID to load the model onto.
            torch_dtype (torch.dtype): Data type for model weights.
            batch_size (int): Batch size for processing inputs.
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.device = device

        # Initialize the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._initialize_token_ids()

        # Load the model ONCE and store it.
        self.model = self.load_model(self.device)

        # Create a lock to guard model inference if needed.
        self.model_lock = threading.Lock()

    def _initialize_token_ids(self):
        """Precompute token IDs for the two labels: 'Yes' and 'No'."""
        responses = ["Yes", "No"]
        token_ids = [
            self.tokenizer(response, add_special_tokens=False)["input_ids"]
            for response in responses
        ]
        # Assume each response is encoded as a single token.
        self.applicable_token_id, self.not_applicable_token_id = [
            ids[0] for ids in token_ids
        ]

    def load_model(self, device):
        """
        Loads the model on the specified device using 4-bit quantization.
        """
        print(f"Loading model on device cuda:{device}...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            quantization_config=quant_config,
            device_map=f"cuda:{device}",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)

        model.eval()
        return model

    def preprocess_text(self, text):
        """
        Normalizes and cleans input text.
        """
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess_pairs(self, patient_trial_pairs):
        """
        Preprocesses patient-trial pairs.
        """
        processed = []
        for patient, trial in patient_trial_pairs:
            patient_clean = self.preprocess_text(patient)
            trial_clean = self.preprocess_text(trial)
            processed.append((patient_clean, trial_clean))
        return processed

    def create_messages(self, patient_text, trial_text):
        """
        Creates a chat-style prompt with a system instruction and a few-shot example.

        The answer must be exactly one of the two labels.
        """
        system_prompt = (
            "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
            "provides enough details to evaluate whether the patient satisfies or violates the clinical "
            "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
            "information to make this evaluation, or 'No' if it does not."
        )
        input_example = {
            "role": "user",
            "content": f"Statement A: {patient_text}\nStatement B: {trial_text}\n\n",
        }
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": " "},
            input_example,
        ]
        return messages

    def process_batch(self, batch):
        """
        Processes a single batch of input pairs using the preloaded model.
        """
        batch_prompts = []
        for patient_text, trial_text in batch:
            messages = self.create_messages(patient_text, trial_text)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(prompt)

        # Tokenize the batch.
        inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Ensure thread-safe access to the model.
        with self.model_lock:
            with torch.no_grad():
                outputs = self.model(**inputs)

        # Compute probabilities from logits.
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        probabilities = F.softmax(last_token_logits, dim=-1)
        applicable_probs = probabilities[:, self.applicable_token_id].tolist()

        batch_results = [
            {"llm_score": prob, "answer": "Yes" if prob > 0 else "No"}
            for prob in applicable_probs
        ]
        return batch_results

    def rank_pairs(self, patient_trial_pairs):
        """
        Classifies each (patient_text, trial_text) pair using the shared model.
        Batches are processed concurrently using threads.
        """
        preprocessed_pairs = self.preprocess_pairs(patient_trial_pairs)

        # Split data into batches.
        batches = [
            preprocessed_pairs[i : i + self.batch_size]
            for i in range(0, len(preprocessed_pairs), self.batch_size)
        ]

        results = []
        # Use ThreadPoolExecutor to process batches concurrently.
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                results.extend(future.result())
        return results
