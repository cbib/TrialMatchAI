import logging
import os
from pathlib import Path

from transformers import HfArgumentParser
from arguments import ModelArguments

# Check if we need to import Unsloth first
early_parser = HfArgumentParser(ModelArguments)
model_args, _ = early_parser.parse_known_args()

if model_args.use_unsloth is True:
    import unsloth

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, set_seed
from arguments import DataArguments, SFTTrainingArguments as TrainingArguments
from data import TrainDataset, DataCollatorForFinetuning
from modeling import LanguageModelFinetuner
from trainer import SFTTrainer
from load_model import get_model


logger = logging.getLogger(__name__)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
# Initialize the distributed environment if needed
dist.init_process_group(backend=os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"))
torch.cuda.set_device(local_rank)

# Get the rank of the current process
rank = dist.get_rank()

# Map the rank to a specific GPU
device_id = rank  # This assumes rank maps to GPU ID
torch.cuda.set_device(device_id)

print(f"Rank {rank} using device {device_id} on {torch.cuda.get_device_name(device_id)}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    base_model = get_model(model_args, training_args)

    if model_args.use_unsloth is True:
        logger.info('Using Unsloth for finetuning...')
        from unsloth import FastLanguageModel
        from peft import LoraConfig

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=model_args.model_query_max_len + model_args.model_passage_max_len,
            dtype=torch.bfloat16 if training_args.bf16 is True else torch.float16,
            load_in_4bit=True,
        )

        lora_config = LoraConfig(
            r=int(model_args.lora_rank),
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.target_modules,
        )
        lora_config_dict = {k: v for k, v in vars(lora_config).items() if not k.startswith('_')}
        logger.info(lora_config_dict)
        if "task_type" in lora_config_dict:
            lora_config_dict.pop("task_type")


        model = FastLanguageModel.get_peft_model(model, **lora_config_dict)

        
    else:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=not model_args.use_slow_tokenizer,
            trust_remote_code=True,
            token=model_args.token
        )

        # Ensure pad_token_id is defined
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                # As a fallback if the tokenizer doesn't have unk_token_id, set pad_token_id to a known token
                # If using a special tokenizer, make sure to adapt accordingly.
                tokenizer.pad_token_id = tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
        logger.info('Config: %s', config)

        model = LanguageModelFinetuner(model=base_model, tokenizer=tokenizer, train_batch_size=training_args.per_device_train_batch_size)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load the training dataset
    train_dataset = TrainDataset(args=data_args, tokenizer=tokenizer)

    # Setup data collator
    data_collator = DataCollatorForFinetuning(
        tokenizer=tokenizer,
        query_max_len=data_args.data_query_max_len,
        passage_max_len=data_args.data_passage_max_len,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.use_lora = model_args.use_lora

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()

    # If not using LoRA, you can save a final checkpoint if desired
    if not model_args.use_lora and trainer.deepspeed is not None:
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        trainer.deepspeed.save_checkpoint(checkpoint_dir)

    # If world process zero, save tokenizer
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all processes finalize
        dist.barrier()  # Synchronize all processes
        dist.destroy_process_group()  # Clean up
