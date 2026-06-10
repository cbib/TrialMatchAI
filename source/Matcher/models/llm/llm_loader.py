from typing import Tuple

import torch
from Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = setup_logging(__name__)


def load_model_and_tokenizer(
    model_config: dict, device: int
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer. CUDA gets 4-bit quant; MPS gets bfloat16; CPU gets float32."""
    use_cuda = torch.cuda.is_available()
    use_mps = not use_cuda and torch.backends.mps.is_available()
    quant_config = None
    attn_impl = None
    compute_dtype = torch.float32

    if use_cuda:
        cuda_count = torch.cuda.device_count()
        idx = int(device) if isinstance(device, int) else 0
        if idx < 0 or idx >= cuda_count:
            logger.warning(
                f"Requested CUDA device {device} invalid; using 0 (num_gpus={cuda_count})."
            )
            idx = 0
        try:
            torch.cuda.set_device(idx)
        except Exception as e:
            logger.warning(f"torch.cuda.set_device({idx}) failed: {e}. Falling back to 0.")
            idx = 0
            torch.cuda.set_device(idx)
        device_str = f"cuda:{idx}"
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            major, minor = torch.cuda.get_device_capability(idx)
            if (major * 10 + minor) >= 75:
                attn_impl = "flash_attention_2"
                logger.info("Using FlashAttention-2.")
            else:
                logger.info("FlashAttention-2 unsupported on this GPU; using SDPA.")
        except Exception:
            logger.info("flash-attn not available; using SDPA.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_config["quantization"]["load_in_4bit"]),
            bnb_4bit_use_double_quant=bool(
                model_config["quantization"]["bnb_4bit_use_double_quant"]
            ),
            bnb_4bit_quant_type=str(model_config["quantization"]["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"Loading model on {device_str} with 4-bit quantization.")
    elif use_mps:
        device_str = "mps"
        compute_dtype = torch.bfloat16
        logger.info("Loading model on MPS (Apple Silicon) in bfloat16.")
    else:
        device_str = "cpu"
        compute_dtype = torch.float32
        logger.warning("CUDA and MPS not available; loading model on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["base_model"],
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            device_map=device_str,
            attn_implementation=attn_impl,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
    else:
        # MPS and CPU: load without device_map (not well-supported), then move
        model = AutoModelForCausalLM.from_pretrained(
            model_config["base_model"],
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device_str)

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if use_cuda:
        model = PeftModel.from_pretrained(
            model, model_config["cot_adapter_path"], device_map=device_str
        )
    else:
        model = PeftModel.from_pretrained(model, model_config["cot_adapter_path"])
        model = model.to(device_str)

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
