import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput


logger = logging.getLogger(__name__)

@dataclass
class LMOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None

class LanguageModelFinetuner(nn.Module):
    def __init__(self, 
             model: nn.Module, 
             tokenizer: AutoTokenizer = None, 
             train_batch_size: int = 4,
             enable_gradient_checkpointing: bool = False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size

        if self.model.config.pad_token_id is None and self.tokenizer is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()


    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads(**kwargs)

    def forward(self, 
                input_ids: torch.Tensor = None, 
                attention_mask: torch.Tensor = None, 
                labels: torch.Tensor = None) -> LMOutput:
        device = next(self.model.parameters()).device  # Move inputs to the model's device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return LMOutput(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits if hasattr(outputs, 'logits') else None
        )

    def save(self, output_dir: str):
        # Save the model (with weights) to output_dir
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, **kwargs):
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(**kwargs)
        return self.model.save_pretrained(**kwargs)
