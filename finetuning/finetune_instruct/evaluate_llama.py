import os
import json
import torch
import transformers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class LlamaMatchMismatchEvaluator:
    """
    Evaluates a Llama-based instruction model on a classification task:
    ('Match', 'Mismatch', or 'Neutral').

    Each data entry is expected to contain:
      {
        "instruction": <string with additional guidance or context>,
        "input": <string with patient/trial statements to classify>,
        "output": <one of "Match", "Mismatch", "Neutral">
      }
    """

    def __init__(
        self,
        model_id: str = "ContactDoctor/Bio-Medical-Llama-3-8B",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto"
    ):
        """
        Initialize the evaluator with the specified Llama-based model.

        :param model_id: Hugging Face model ID (must be Llama-based).
        :param torch_dtype: Floating-point precision (bfloat16 recommended on GPUs like A100).
        :param device_map: Device mapping setting for large models, e.g. "auto".
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        # Create a text-generation pipeline for Llama
        self.pipe = transformers.pipeline(
            task="text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device_map,
            trust_remote_code=True
        )

        # Grab tokenizer from the pipeline (useful for custom logic)
        self.tokenizer = self.pipe.tokenizer

        # Define a system prompt for the conversation:
        # (Customizable to remind the model of its domain)
        self.system_prompt = (
            "You are a clinical trials expert specializing in matching patients "
            "to eligibility criteria. You must respond with exactly one of: "
            "'Match', 'Mismatch', or 'Neutral'."
        )

    def _build_prompt(self, data_entry):
        """
        Builds a single prompt (string) that includes both system and user instructions.
        Uses apply_chat_template to emulate a conversation.
        
        :param data_entry: A dict with keys: "instruction", "input".
        :return: A single text prompt ready for the pipeline.
        """
        # Format role-based messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{data_entry['instruction']}\n\n"
                           f"{data_entry['input']}\n\n"
                           "Respond with 'Match', 'Mismatch', or 'Neutral' only."
            }
        ]

        # Llama's apply_chat_template for multi-turn chat
        prompt = self.tokenizer.apply_chat_template(
            [messages],  # pass a nested list of conversation turns
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt

    def _extract_class_label(self, generated_text: str) -> str:
        """
        Attempts to extract the predicted label from the model's output text.
        Returns 'Match', 'Mismatch', or 'Neutral' if found; otherwise 'Unknown'.

        :param generated_text: The raw string output from the pipeline.
        :return: One of "Match", "Mismatch", "Neutral", or "Unknown".
        """
        # Simple case-insensitive substring check. You can refine parsing if needed.
        for label in ["Match", "Mismatch", "Neutral"]:
            if label.lower() in generated_text.lower():
                return label
        return "Unknown"

    def generate_predictions(
        self,
        dataset: list,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> list:
        """
        Generates label predictions for each dataset entry using the pipeline.

        :param dataset: List of examples, each with "instruction", "input".
        :param max_new_tokens: Maximum tokens to generate for each inference call.
        :param temperature: Temperature for sampling. 0.0 = deterministic.
        :param top_p: Nucleus sampling parameter (1.0 = no nucleus sampling).
        :return: List of predicted labels (strings).
        """
        predictions = []
        for data_entry in dataset:
            # Build the prompt from system + user messages
            prompt = self._build_prompt(data_entry)

            # Define custom end-of-sequence tokens if needed.
            # Some Llama-based models use custom tokens or just rely on eos_token_id.
            eos_token_ids = [
                self.tokenizer.eos_token_id,
                # If your model uses <|end_of_turn|> or similar, convert to ID:
                # self.tokenizer.convert_tokens_to_ids("<|end_of_turn|>")
            ]

            # Run inference. We get a list of generated sequences; pick the first.
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_ids,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p
            )
            # The pipeline returns a list of dicts. We extract the full text from the first.
            raw_generated = outputs[0]["generated_text"]

            # Slice out only the newly generated portion if desired.
            # The prompt is the first part; everything after that is the response.
            response_text = raw_generated[len(prompt) :]

            # Extract class label from the response
            predicted_label = self._extract_class_label(response_text)
            predictions.append(predicted_label)

        return predictions

    def evaluate(self, dataset: list) -> dict:
        """
        Generates predictions and computes the classification metrics.

        :param dataset: List of examples, each with "output" as the ground truth label.
        :return: Dictionary with accuracy, precision, recall, and f1 (weighted).
        """
        # Ground truth labels
        references = [entry["output"] for entry in dataset]
        # Predictions
        predictions = self.generate_predictions(dataset)

        # Compute metrics
        accuracy = accuracy_score(references, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average="weighted", labels=["Match", "Mismatch", "Neutral"]
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


if __name__ == "__main__":
    # Example usage:

    # 1. Load a dataset from JSONL or any other source.
    #    Each line should have keys: 'instruction', 'input', 'output'
    #    where 'output' is one of "Match", "Mismatch", "Neutral".
    dataset_path = "cleaned_filtered_test_data.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # 2. Initialize the evaluator
    evaluator = LlamaMatchMismatchEvaluator(
        model_id="ContactDoctor/Bio-Medical-Llama-3-8B",
        torch_dtype=torch.bfloat16,  # or torch.float16
        device_map="auto"
    )

    # 3. Run evaluation
    results = evaluator.evaluate(dataset)

    # 4. Print results
    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
