import os
import torch
import torch.nn.functional as F
import unicodedata
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from multiprocessing import get_context
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import json
import random

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class EvaluationScript:
    def __init__(self, model_path, adapter_path=None, devices=None, torch_dtype=torch.float16):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.devices = devices or list(range(torch.cuda.device_count()))

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Initialize token IDs for responses
        self._initialize_token_ids()

    def _initialize_token_ids(self):
        responses = ["Match", "Mismatch", "Neutral"]
        token_ids = [self.tokenizer(response, add_special_tokens=False)["input_ids"] for response in responses]
        self.match_token_id, self.mismatch_token_id, self.neutral_token_id = [ids[0] for ids in token_ids]

    def load_model(self, device):
        print(f"Loading model on device cuda:{device}...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            quantization_config=quant_config,
            device_map=f"cuda:{device}",
            trust_remote_code=True,
        )
        model.eval()
        return model

    def preprocess_text(self, text):
        match_a = re.search(r"Statement A:\s*(.+?)(?=\nStatement B:|$)", text, re.DOTALL)
        match_b = re.search(r"Statement B:\s*(.+)", text, re.DOTALL)

        statement_a = unicodedata.normalize("NFKD", match_a.group(1).strip()) if match_a else ""
        statement_b = unicodedata.normalize("NFKD", match_b.group(1).strip()) if match_b else ""
        
        # remove unecessary punctuations
        statement_a = re.sub(r'[^\w\s]', '', statement_a)
        statement_b = re.sub(r'[^\w\s]', '', statement_b)

        combined_text = f"Statement A: {statement_a}\nStatement B: {statement_b}"

        return combined_text

    def create_messages(self, input_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable AI medical assistant. Your task is to evaluate the semantic consistency "
                    "and logical alignment between Statement A (patient description) and Statement B (eligibility criterion). "
                    "Apply the following rules strictly:\n"
                    "- Reply 'Match' if Statement A aligns with and satisfies Statement B.\n"
                    "- Reply 'Mismatch' if Statement A contradicts or is incompatible with Statement B.\n"
                    "- Reply 'Neutral' if Statement A and Statement B are unrelated, loosely related, or ambiguously connected "
                    "without clear alignment or contradiction.\n\n"
                    "Provide your response using only 'Match,' 'Mismatch,' or 'Neutral'."
                ),
            },
            # Few-shot example 1 - Match
            {
                "role": "user",
                "content": (
                    "Statement A: The patient has a confirmed diagnosis of Stage IV non-small cell lung cancer (NSCLC) "
                    "with an EGFR exon 19 deletion mutation.\n"
                    "Statement B: NSCLC cases associated with EGFR exon 19 mutations."
                ),
            },
            {
                "role": "assistant",
                "content": "Match",
            },
            # Few-shot example 2 - Mismatch
            {
                "role": "user",
                "content": (
                    "Statement A: The patient has a history of allergic reactions to penicillin.\n"
                    "Statement B: No known severe hypersensitivity reactions to study drugs, especially to penicillin."
                ),
            },
            {
                "role": "assistant",
                "content": "Mismatch",
            },
            # Few-shot example 3 - Neutral
            {
                "role": "user",
                "content": (
                    "Statement A: The patient is a 65-year-old male with mild osteoarthritis managed with physical therapy.\n"
                    "Statement B: Patients with respiratory disorders currently attending rehabilition and physical therapy."
                ),
            },
            {
                "role": "assistant",
                "content": "Neutral",
            },
            # Input example to evaluate
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

            input_text = self.preprocess_text(input_data["input"])
            messages = self.create_messages(input_text)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(f"cuda:{device}")

            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            last_token_logits = outputs.logits[0, -1, :]
            probabilities = torch.softmax(last_token_logits, dim=-1)
            top_prob, top_token_id = torch.topk(probabilities, 1)
            top_token = self.tokenizer.decode(top_token_id[0])
            prob = top_prob.item()

            if top_token_id == self.match_token_id:
                answer = "Match"
            elif top_token_id == self.mismatch_token_id:
                answer = "Mismatch"
            elif top_token_id == self.neutral_token_id:
                answer = "Neutral"
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

        # with open(save_path, "w", encoding="utf-8") as f:
        #     for example in incorrect_examples:
        #         json.dump(example, f)
        #         f.write("\n")

        references = [entry["output"] for entry in dataset]
        predictions_only = [pred["prediction"] for pred in predictions]

        accuracy = accuracy_score(references, predictions_only)
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions_only, average="weighted"
        )

        print(f"Incorrect examples saved to {save_path}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

if __name__ == "__main__":
    model_path = "tiiuae/Falcon3-10B-Instruct"
    adapter_path = None
    devices = [0, 1]

    evaluator = EvaluationScript(
        model_path=model_path,
        adapter_path=adapter_path,
        devices=devices,
        torch_dtype=torch.float16
    )
    dataset_path = "finetuning_data/cleaned_filtered_test_data.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
        dataset = random.sample(dataset, 100)
#     dataset = [
# {"input": "Statement A: no history of liver diseases or kidney disease is present, and serum creatinine levels are maintained below 1.0 mg/dl.\nStatement B: participants must not have a history of liver diseases (such as hepatitis, biliary atresia, or cirrhosis) or kidney disease, defined by a serum creatinine level exceeding 1.0 mg/dl.", "output": "Match"}

#     ]
    results = evaluator.evaluate(dataset)

    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
