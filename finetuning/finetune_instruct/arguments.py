import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments

# Set desired GPU indices if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"


def default_list() -> List[str]:
    return ["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_model_path: str = field(default='')
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank dimension for LoRA."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The scaling factor (alpha) for LoRA."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout rate for LoRA layers."}
    )
    target_modules: List[str] = field(
        default_factory=default_list,
        metadata={"help": "List of modules to apply LoRA to."}
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If True, merges the LoRA parameters into the base model before saving."}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If True, use flash attention during training (if supported)."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If True, use a slow (Python-based) tokenizer instead of a fast (C++/Rust) one."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "If True, create the model as an empty shell and then load weights, reducing RAM usage."}
    )
    cache_dir: str = field(
        default="tmp", 
        metadata={"help": "Path to the directory where models and tokenizers are cached."}
    )
    token: str = field(
        default=None, 
        metadata={"help": "HuggingFace hub token for private models."}
    )
    from_peft: str = field(
        default=None,
        metadata={"help": "Path to a PEFT checkpoint from which to load a model."}
    )
    lora_extra_parameters: str = field(
        default=None,
        metadata={"help": "Additional modules to save when using LoRA."}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default='toy_finetune_data.jsonl', 
        metadata={"help": "Path to the training data file (in JSONL format)."}
    )

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "Max length of the input sequence for the instruction/input portion."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "Max length of the entire sequence (instruction + input + output)."
        },
    )

    max_example_num_per_dataset: int = field(
        default=10, 
        metadata={"help": "Maximum number of examples to load from the dataset."}
    )
    
    cache_path: str = field(
        default='./data_dir',
        metadata={"help": "Directory for caching processed datasets."}
    )

    load_from_disk: bool = field(
        default=False, 
        metadata={"help": "If True, load a previously saved dataset from disk instead of processing from scratch."}
    )

    load_disk_path: str = field(
        default=None, 
        metadata={"help": "Path to the saved dataset on disk if load_from_disk is True."}
    )

    save_to_disk: bool = field(
        default=False, 
        metadata={"help": "If True, save the processed dataset to disk."}
    )

    save_disk_path: str = field(
        default=None, 
        metadata={"help": "Path to save the processed dataset if save_to_disk is True."}
    )

    num_shards: int = field(
        default=0, 
        metadata={"help": "Number of shards to split the dataset into when saving."}
    )

    save_max_shard_size: str = field(
        default="50GB", 
        metadata={"help": "Maximum size of each shard when saving the dataset."}
    )

    exit_after_save: bool = field(
        default=False, 
        metadata={"help": "If True, exit the program after saving the dataset."}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"Cannot find file: {self.train_data}. Please provide a valid path.")


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """
    Training arguments specifically for supervised fine-tuning a causal language model.
    """
    # Additional arguments can be added if needed.
    pass
