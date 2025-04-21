import re
import sys
import os
import random
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, DataCollatorForSeq2Seq

class TrainDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer
    ):
        # Load dataset (assuming args.train_data points to a directory or a JSON file)
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                try:
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                         split='train',
                                                         cache_dir=args.cache_path)
                except Exception as e:
                    print(e, file)
                    sys.exit()
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(range(len(temp_dataset)), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)

            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train', cache_dir=args.cache_path)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        # Define prompt format
        messages = [
            {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!{instruction}"},
            {"role": "user", "content": "{input}\n"},
        ]
        self.prompt_format = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Maximum length for the model input
        self.max_length = self.args.query_max_len + self.args.passage_max_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        example = self.dataset[index]
        instruction = example['instruction']
        input_text = example.get('input', "")
        output_text = example['output']

        # Create the prompt
        prompt = self.prompt_format.format(
            instruction=instruction.strip(),
            input=input_text.strip()
        )

        # Tokenize the prompt and output together
        full_input = prompt + output_text
        tokenized = self.tokenizer(
            full_input,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Determine where output starts
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )
        prompt_len = len(prompt_tokenized['input_ids'])

        # Create labels
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
    """
    Collator that pads input_ids, attention_masks, and labels for LLaMA fine-tuning.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    label_pad_token_id: int = -100
    tokenizer: PreTrainedTokenizerBase = None
    padding: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to the DataCollator.")
        if self.tokenizer.pad_token_id != 128009:
            raise ValueError("The tokenizer pad_token_id must be set to 128009 for LLaMA.")

    def __call__(self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None) -> Dict[str, Any]:
        return_tensors = return_tensors or self.return_tensors

        # Separate labels
        labels = [feature.pop("labels") for feature in features]

        # Pad labels
        label_features = [{"input_ids": label} for label in labels]
        padded_labels = self.tokenizer.pad(
            label_features,
            padding="longest",
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels_tensor = padded_labels["input_ids"]
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = self.label_pad_token_id

        # Pad input_ids and attention_mask
        max_length = self.query_max_len + self.passage_max_len
        padded_features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Add labels to the padded features
        padded_features["labels"] = labels_tensor
        return padded_features
