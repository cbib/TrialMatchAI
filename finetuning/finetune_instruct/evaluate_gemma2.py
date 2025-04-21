import os
import torch
import unicodedata
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from multiprocessing import get_context
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from peft import PeftModel
import json
import random

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class EvaluationScript:
    def __init__(self, model_path, adapter_path=None, devices=None, torch_dtype=torch.float16):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.devices = devices or list(range(torch.cuda.device_count()))
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize token IDs for responses "Yes" and "No"
        self._initialize_token_ids()
        
    def _initialize_token_ids(self):
        responses = ["Yes", "No"]
        token_ids = [self.tokenizer(response, add_special_tokens=False)["input_ids"] for response in responses]
        self.yes_token_id, self.no_token_id = [ids[0] for ids in token_ids]
        
    def load_model(self, device):
        """
        Loads the model on the specified device using 4-bit quantization.
        If an adapter path is provided, it loads the adapter using PEFT.
        """
        print(f"Loading model on device cuda:{device}...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=f"cuda:{device}",
            attn_implementation="flash_attention_2",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        if self.adapter_path is not None:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return model

    def preprocess_text(self, text):
        # Extract Statement A and Statement B using regex.
        match_a = re.search(r"Statement A:\s*(.+?)(?=\nStatement B:|$)", text, re.DOTALL)
        match_b = re.search(r"Statement B:\s*(.+)", text, re.DOTALL)
        
        statement_a = unicodedata.normalize("NFKD", match_a.group(1).strip()) if match_a else ""
        statement_b = unicodedata.normalize("NFKD", match_b.group(1).strip()) if match_b else ""
        
        # Optionally, remove unnecessary punctuations
        statement_a = re.sub(r'[^\w\s]', '', statement_a)
        statement_b = re.sub(r'[^\w\s]', '', statement_b)
        
        combined_text = f"Statement A: {statement_a}\nStatement B: {statement_b}"
        return combined_text

    def create_messages(self, input_text):
        # Create a prompt with only the system instruction and the input example.
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
                    "is related to the "
                    "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
                    "information to make this evaluation, or 'No' if it does not."
                ),
            },
            {"role": "assistant", "content": " "},
            {
                "role": "user",
                "content": input_text,
            },
        ]
        return messages

    def process_on_device(self, device, input_queue, output_queue):
        model = self.load_model(device)
        
        while True:
            input_data = input_queue.get()
            if input_data is None:
                break
            
            # Preprocess the input text (extract and clean Statement A and B)
            input_text = self.preprocess_text(input_data["input"])
            messages = self.create_messages(input_text)
            
            # Use the model's chat template if available to format the prompt.
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(f"cuda:{device}")
            
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
            
            # Get the logits of the last token and determine the top prediction.
            last_token_logits = outputs.logits[0, -1, :]
            probabilities = torch.softmax(last_token_logits, dim=-1)
            top_prob, top_token_id = torch.topk(probabilities, 1)
            prob = top_prob.item()
            
            if top_token_id == self.yes_token_id:
                answer = "Yes"
            elif top_token_id == self.no_token_id:
                answer = "No"
            else:
                answer = "Unknown"
            
            result = {
                "input": input_data["input"],
                "ground_truth": input_data["output"],
                "prediction": answer,
                "llm_score": prob,
            }
            output_queue.put(result)
    
    def generate_predictions(self, dataset):
        ctx = get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        
        workers = [
            ctx.Process(
                target=self.process_on_device,
                args=(device, input_queue, output_queue)
            )
            for device in self.devices
        ]
        
        for worker in workers:
            worker.start()
        
        for entry in dataset:
            input_queue.put(entry)
        
        # Signal workers to exit
        for _ in self.devices:
            input_queue.put(None)
        
        all_predictions = []
        for _ in tqdm(range(len(dataset)), desc="Processing inputs"):
            all_predictions.append(output_queue.get())
        
        for worker in workers:
            worker.join()
        
        return all_predictions
    
    def evaluate(self, dataset, save_path="incorrect_predictions.jsonl"):
        predictions = self.generate_predictions(dataset)
        
        incorrect_examples = [
            {
                "input": entry["input"],
                "ground_truth": entry["output"],
                "prediction": pred["prediction"],
                "llm_score": pred["llm_score"]
            }
            for entry, pred in zip(dataset, predictions)
            if pred["prediction"] != entry["output"]
        ]
        
        # Optionally, save incorrect examples.
        # with open(save_path, "w", encoding="utf-8") as f:
        #     for example in incorrect_examples:
        #         json.dump(example, f)
        #         f.write("\n")
        
        references = [entry["output"] for entry in dataset]
        predictions_only = [pred["prediction"] for pred in predictions]
        
        accuracy = accuracy_score(references, predictions_only)
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions_only, average="weighted", zero_division=0
        )
        
        print(f"Incorrect examples saved to {save_path}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

if __name__ == "__main__":
    # Use the model from the specified path.
    model_path = "google/gemma-2-2b-it"
    # Set adapter_path to a valid path if you want to load a PEFT adapter; otherwise, keep it as None.
    adapter_path = "finetuned_gemma2/"
    devices = [0]  # Adjust the devices as needed.
    
    evaluator = EvaluationScript(
        model_path=model_path,
        adapter_path=adapter_path,
        devices=devices,
        torch_dtype=torch.float16
    )
    
    # Path to the transformed test dataset.
    dataset_path = "finetuning_data/transformed_test_data.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
        # Optionally, sample a subset for evaluation.
        # dataset = random.sample(dataset, 100)
    
    results = evaluator.evaluate(dataset)
    
    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
