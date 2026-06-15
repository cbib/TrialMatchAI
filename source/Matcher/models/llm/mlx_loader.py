import os
from typing import Tuple

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def load_mlx_model(model_config: dict) -> Tuple:
    """
    Load phi-4 (or any HF model) via mlx-lm for Apple Silicon inference.

    If a merged model cache exists at mlx_merged_model_path, that is loaded
    directly (fast).  Otherwise the base model is loaded from HuggingFace.

    LoRA adapter note
    -----------------
    mlx-lm's adapter_path expects adapters trained with mlx_lm.lora, not
    HuggingFace PEFT.  To use the fine-tuned adapter, merge it once:

        python -c "
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        base = AutoModelForCausalLM.from_pretrained('microsoft/phi-4',
                   torch_dtype=torch.float16)
        m = PeftModel.from_pretrained(base, 'models/finetuned_phi_reasoning')
        m.merge_and_unload().save_pretrained('models/merged_phi4_mlx')
        AutoTokenizer.from_pretrained('microsoft/phi-4').save_pretrained(
                   'models/merged_phi4_mlx')
        "

    Then set  \"mlx_merged_model_path\": \"models/merged_phi4_mlx\"  in config.json.
    """
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        raise ImportError(
            "mlx-lm is not installed. Run: pip install mlx-lm"
        )

    base_model = model_config["base_model"]
    merged_path = model_config.get("mlx_merged_model_path", "")

    if merged_path and os.path.isdir(merged_path):
        load_path = merged_path
        logger.info("Loading merged MLX model from %s", load_path)
    else:
        load_path = base_model
        if merged_path:
            logger.warning(
                "mlx_merged_model_path '%s' not found — loading base model without adapter. "
                "See mlx_loader.py docstring to create the merged model.",
                merged_path,
            )
        else:
            logger.info(
                "Loading base MLX model %s (no adapter). "
                "Set mlx_merged_model_path in config.json to use fine-tuned weights.",
                load_path,
            )

    model, tokenizer = mlx_load(load_path)  # specifically for Apple Silicon inference
    logger.info("MLX model ready.")
    return model, tokenizer
