import os
import torch
import logging
from typing import Optional

from transformers.trainer import Trainer
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)

class SFTTrainer(Trainer):
    use_lora: bool

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Custom saving logic depending on whether we're using LoRA or not
        if not self.use_lora:
            # If not using LoRA, just use the default save implementation
            super()._save(output_dir, state_dict)
            return

        # Using LoRA
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Ensure model has the `save` method implemented
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} does not support save interface'
            )
        else:
            self.model.save(output_dir)

        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # If using DeepSpeed ZeRO-3, save LoRA adapters separately
        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'model.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                logger.info(f"Saved LoRA adapter model at {output_dir}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer.
        For causal language modeling tasks, the model returns the loss directly if labels are provided.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Optionally use num_items_in_batch if needed for custom logic
        if num_items_in_batch is not None:
            # Example: Adjust loss based on batch size if required
            pass

        return (loss, outputs) if return_outputs else loss
