"""Merge a LoRA adapter into its base model to produce a standalone checkpoint.

vLLM serves LoRA adapters natively (via LoRARequest), so merging is optional.
Use it when you prefer a single self-contained model directory over base+adapter.
"""

from __future__ import annotations

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    *,
    trust_remote_code: bool = False,
) -> str:
    """Load base + LoRA adapter, merge weights, and save a full model to output_dir."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Merging requires the optional `finetune` dependencies "
            "(`uv sync --extra finetune`)."
        ) from exc

    logger.info("Loading base model %s ...", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    logger.info("Applying and merging adapter %s ...", adapter_path)
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code).save_pretrained(
        output_dir
    )
    logger.info("Saved merged model to %s", output_dir)
    return output_dir
