from typing import Any, Tuple

from trialmatchai.models.llm._common import (
    build_4bit_quant_config,
    configure_decoder_tokenizer,
    load_llm_dependencies,
    resolve_cuda_device,
    select_attn_impl,
    select_compute_dtype,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def load_model_and_tokenizer(model_config: dict, device: int) -> Tuple[Any, Any]:
    """Load a model and tokenizer with safe device handling and optional 4-bit."""
    deps = load_llm_dependencies()
    torch = deps.torch

    device_str, cuda_index = resolve_cuda_device(
        torch, device, label="load_model_and_tokenizer"
    )
    use_cuda = cuda_index is not None
    compute_dtype = select_compute_dtype(torch, use_cuda)
    attn_impl = select_attn_impl(torch, cuda_index)

    if use_cuda:
        quant_config = build_4bit_quant_config(
            deps.bnb_config,
            load_in_4bit=bool(model_config["quantization"]["load_in_4bit"]),
            double_quant=bool(
                model_config["quantization"]["bnb_4bit_use_double_quant"]
            ),
            quant_type=str(model_config["quantization"]["bnb_4bit_quant_type"]),
            compute_dtype=compute_dtype,
        )
        logger.info(f"Loading model on {device_str} with 4-bit quantization.")
    else:
        logger.warning(
            "CUDA not available; loading model on CPU without 4-bit quantization."
        )
        quant_config = build_4bit_quant_config(deps.bnb_config, load_in_4bit=False)

    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    revision = model_config.get("base_model_revision")
    tokenizer = deps.auto_tokenizer.from_pretrained(
        model_config["base_model"],
        revision=revision,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    # Always left-pad decoder-only models; keep most recent tokens on truncation.
    configure_decoder_tokenizer(tokenizer)

    model = deps.auto_model.from_pretrained(
        model_config["base_model"],
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=compute_dtype,
        device_map=device_str,
        attn_implementation=attn_impl,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    # Ensure KV cache usage for faster generation
    try:
        model.config.use_cache = True
    except Exception:
        pass

    model = deps.peft_model.from_pretrained(
        model, model_config["cot_adapter_path"], device_map=device_str
    )

    # Optional: compile for extra speed when supported
    if bool(model_config.get("compile", False)):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed; continuing without it. Err: {e}")

    if isinstance(model, torch.nn.Module):
        model.eval()
    else:
        logger.warning("Model is not an instance of torch.nn.Module; skipping eval.")
    logger.info(f"Model loaded on {device_str}.")
    return model, tokenizer  # type: ignore[return-value]
