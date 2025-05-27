import os
import sys
import re
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import datasets
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    BatchEncoding,
)
from jinja2 import Template

# Replace this import with your own arguments class or define it below
from arguments import DataArguments

instruction_template = r"""
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
<bos>
{% for message in messages %}
{% if loop.first and message['role'] == 'system' %}
    {% set role = 'user' %}
{% elif message['role'] == 'assistant' %}
    {% set role = 'model' %}
{% else %}
    {% set role = message['role'] %}
{% endif %}
<start_of_turn>{{ role }}
{{ message['content'].rstrip() }}<end_of_turn>
{% endfor %}
"""

def apply_chat_template(messages, add_generation_prompt=True):
    """
    Renders a list of messages (dicts with 'role' and 'content') into
    a single text prompt for causal language models, using Jinja.
    """
    t = Template(instruction_template)
    rendered = t.render(messages=messages)
    if add_generation_prompt:
        # Optionally append a small trigger or instruction for the model to continue
        rendered += "\n<start_of_turn>user\n"
    return rendered



class TrainDataset(Dataset):
    """
    A PyTorch Dataset for instruction-tuning on a causal language model.
    """
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        """
        :param args: Custom DataArguments containing dataset paths, cache paths, etc.
        :param tokenizer: A pretrained tokenizer (GPT-2, GPT-NeoX, or similar).
        """
        # Load the dataset
        self.dataset = datasets.load_dataset(
            'json',
            data_files=args.train_data,
            split='train',
            cache_dir=args.cache_path
        )
        
        # Shuffle the dataset using random 
        self.dataset = self.dataset.shuffle(seed=42)
        

        # Randomly sample specified number of examples if needed
        if len(self.dataset) > args.max_example_num_per_dataset:
            sampled_indices = random.sample(
                range(len(self.dataset)),
                args.max_example_num_per_dataset
            )
            self.dataset = self.dataset.select(sampled_indices)

        # Basic setup
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        # Set the tokenizer's pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Generate the prompt format
        messages = [
            {     
                "role": "system", 
                "content": "{instruction}"
                   
            },
            
            {
                "role": "user",
                "content": (
                    "{input}\n"
                ),
            },
        ]
        self.prompt_format = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("Prompt format:", self.prompt_format)

        # Set max_length for prompt+output
        self.max_length = self.args.data_query_max_len + self.args.data_passage_max_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        For each example:
          1. Read instruction, input, and output from the dataset.
          2. Build the prompt using the stored template.
          3. Concatenate prompt + output_text as the full input.
          4. Tokenize everything. 
          5. Mask out the prompt portion in labels with -100.
        """
        example = self.dataset[index]
        instruction = example['instruction']
        input_text = example.get('input', "")
        output_text = example['output']

        # Build the prompt from our custom format
        prompt = self.prompt_format.format(
            instruction=instruction.strip(),
            input=input_text.strip()
        )
        
        # Concatenate prompt + output in one sequence for causal LM
        full_input = prompt + output_text.strip()

        # Tokenize the entire sequence
        tokenized = self.tokenizer(
            full_input,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,  # return raw python lists
            add_special_tokens=True
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Tokenize just the prompt to find boundary
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )
        prompt_len = len(prompt_tokenized['input_ids'])

        # Create labels array and mask out prompt tokens with -100
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
    Collator that pads input_ids, attention_mask, and labels for a 
    causal instruction-tuning scenario.
    
    In many seq2seq settings (T5/BART), you would rely on DataCollatorForSeq2Seq 
    to handle encoder/decoder inputs. However, here we can still inherit from it 
    for convenience if we only need padding on a single sequence + labels.

    If using a GPT-style model, consider using DataCollatorForLanguageModeling 
    from Hugging Face Transformers. But this approach will still work 
    if you only need to handle padding.
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

    def __call__(self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None) -> Dict[str, Any]:
        """
        Pads inputs and labels separately, then returns a batch of tensors.
        """
        # 1. We use return_tensors from self if not explicitly provided
        return_tensors = return_tensors or self.return_tensors

        # 2. Extract and remove labels from features
        labels = [f.pop("labels") for f in features]

        # 3. Pad labels (treated as input_ids) to the longest label length
        label_features = [{"input_ids": label} for label in labels]
        padded_labels = self.tokenizer.pad(
            label_features,
            padding="longest",
            max_length=None,  # Let tokenizer do max length for labels
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Convert padded label IDs to a torch.Tensor of type long
        labels_tensor = padded_labels["input_ids"]
        if not isinstance(labels_tensor, torch.Tensor):
            labels_tensor = torch.tensor(labels_tensor, dtype=torch.long)

        # Replace pad_token_id with -100 so they don't affect the loss
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = self.label_pad_token_id

        # 4. Pad the input features themselves (input_ids, attention_mask)
        max_length = self.query_max_len + self.passage_max_len
        padded_features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # 5. Add the labels back in
        padded_features["labels"] = labels_tensor

        return padded_features
