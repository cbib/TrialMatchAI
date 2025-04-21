import re
import sys
from typing import List

import math
import os.path
import random
from dataclasses import dataclass
import torch
import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding
from local_gemma import LocalGemma2ForCausalLM

import os
from arguments import DataArguments


class TrainDataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        # Load dataset
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                try:
                    temp_dataset = datasets.load_dataset(
                        'json',
                        data_files=os.path.join(args.train_data, file),
                        split='train',
                        cache_dir=args.cache_path
                    )
                except Exception as e:
                    print(e)
                    print(file)
                    sys.exit()
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset)
                    )
                train_datasets.append(temp_dataset)

            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset(
                'json',
                data_files=args.train_data,
                split='train',
                cache_dir=args.cache_path
            )

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.max_length = self.args.query_max_len + self.args.passage_max_len

    def __len__(self):
        return self.total_len
    
    def apply_chat_template(self, chat_messages):
        """Formats chat messages into a prompt for the model."""
        return (
            "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_messages) +
            "\nRespond with only the JSON object conforming to the schema."
        )

    def __getitem__(self, index):
        example = self.dataset[index]

        # Extract fields from the dataset
        inclusion_criteria = example.get('inclusion_criteria', '').strip()
        exclusion_criteria = example.get('exclusion_criteria', '').strip()
        classification = example.get('classification', '').strip()
        reasoning = example.get('reasoning', '').strip()

        # Universal instruction as in the system chat role
        system_message = (
        "You are a medical assistant tasked with evaluating a patient's eligibility for a clinical trial given the provided eligibility criteria and patient description."
        "Your assessment must be detailed, accurate, and strictly based on the provided patient profile and eligibility criteria."
        "\n\n"
        "### Key Guidelines\n"
        "1. **Critical Information**:\n"
        "   - Essential data includes age, sex, primary diagnosis (e.g., cancer type and stage), and treatment history.\n"
        "   - If any critical information is missing from the patient profile, classify the case as **Excluded**, explaining which data is missing and why it affects the decision.\n\n"
        "2. **Classification Rules**:\n"
        "   - **Included**: The patient meets all inclusion criteria and violates no exclusion criteria.\n"
        "   - **Excluded**: The patient violates any exclusion criterion or fails to meet one or more inclusion criteria.\n"
        "   - **Undetermined**: When critical information is missing, preventing a definitive classification.\n\n"
        "3. **Evaluation Process**:\n"
        "   - Compare each eligibility criterion with the patient's profile.\n"
        "   - Evaluate strictly based on the provided data—do not infer or assume missing details.\n"
        "   - Clearly outline your reasoning, step by step, ensuring it is logically structured and references specific criteria.\n\n"
        "4. **Reasoning Template**:\n"
        "   - Always begin with a summary of the patient's key information relevant to the trial.\n"
        "   - Evaluate inclusion criteria one-by-one:\n"
        "     - Specify whether each criterion is met or not, with evidence from the profile.\n"
        "   - Evaluate exclusion criteria one-by-one:\n"
        "     - Specify whether each criterion is violated or not, with evidence from the profile.\n"
        "   - Conclude with your classification decision, summarizing how inclusion and exclusion criteria were addressed.\n\n"
        "### Response Format\n"
        "Your response must be a JSON object with the following structure:\n"
        "{\n"
        '  "classification": "string",\n'
        '  "reasoning": "string"\n'
        "}\n\n"
        "### Example Reasoning Structure\n"
        "Here is an example reasoning structure:\n\n"
        "1. **Summary**:\n"
        "   The patient is a 65-year-old male with stage IV NSCLC who has completed two prior lines of chemotherapy.\n"
        "2. **Inclusion Criteria Evaluation**:\n"
        "   - Criterion: Patient must have stage IV NSCLC. **Met**: Patient profile confirms stage IV NSCLC.\n"
        "   - Criterion: Must have measurable disease. **Not Evaluated**: Patient profile does not specify this information.\n"
        "3. **Exclusion Criteria Evaluation**:\n"
        "   - Criterion: Prior treatment with immunotherapy. **Violated**: Patient profile confirms prior treatment with immunotherapy.\n"
        "4. **Conclusion**:\n"
        "   Based on the criteria evaluation, the patient is classified as **Excluded** due to violation of the exclusion criterion regarding prior immunotherapy treatment.\n\n"
        "### Additional Notes\n"
        "- Be concise and precise in your reasoning.\n"
        "- Avoid making assumptions about missing or unspecified information.\n"
        "- If multiple factors lead to a classification, clearly state their contributions."
    )

        # Construct the user message with inclusion and exclusion criteria
        user_message = (
            f"**Clinical Trial Eligibility Criteria:**\n\n"
            f"**Inclusion Criteria:**\n{inclusion_criteria}\n\n"
            f"**Exclusion Criteria:**\n{exclusion_criteria}\n\n"
            "Evaluate the compatibility of the patient's profile with the eligibility criteria. Base your classification and reasoning only on the information provided."
        )

        # Build the chat-based prompt
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self.apply_chat_template(chat)

        # Construct the full input with the expected output appended
        full_input = f"{prompt}\n\n{{\n  \"classification\": \"{classification}\",\n  \"reasoning\": \"{reasoning}\"\n}}"

        # Tokenize the full input
        tokenized = self.tokenizer(
            full_input,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Tokenize just the prompt to find where the completion starts
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )
        prompt_len = len(prompt_tokenized['input_ids'])

        # Create labels, masking the prompt portion
        labels = [-100] * len(input_ids)
        if prompt_len < len(input_ids):
            labels[prompt_len:] = input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


@dataclass
class DataCollatorForFinetuning(DataCollatorForSeq2Seq):
    query_max_len: int = 32  # Maximum length for the prompt
    passage_max_len: int = 128  # Maximum length for reasoning and classification

    def __call__(self, features, return_tensors='pt'):
        """
        Custom data collator for fine-tuning a seq2seq model with chat-style prompts and structured output.

        Args:
            features (list): A list of feature dictionaries containing `input_ids`, `attention_mask`, and `labels`.
            return_tensors (str): The format for returned tensors (default is 'pt').
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Extract labels and calculate the maximum label length
        labels = [f["labels"] for f in features]
        max_label_length = max(len(l) for l in labels)

        # Apply padding to labels
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Determine padding side (left or right)
        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # Pad input_ids and attention_mask, ensuring proper alignment with labels
        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.passage_max_len,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Apply masking to prevent training on prompt tokens
        if "labels" in collated:
            collated["labels"] = collated["labels"].clone().detach().to(torch.long)
            collated["labels"][collated["labels"] == self.tokenizer.pad_token_id] = -100  # Mask padding tokens for labels

        return collated
