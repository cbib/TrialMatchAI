"""Shared helpers for the optional LLM model loaders (llm_loader, llm_reranker).

The heavy dependencies (torch/transformers/peft) are imported lazily by
``load_llm_dependencies`` so this module imports cleanly without the ``llm``
extra installed. The torch-dependent helpers take ``torch`` as an argument and
are otherwise pure, which keeps them unit-testable with a stub.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class LLMDeps(NamedTuple):
    torch: Any
    torch_functional: Any
    peft_model: Any
    auto_model: Any
    auto_tokenizer: Any
    bnb_config: Any


def load_llm_dependencies() -> LLMDeps:
    """Import the optional LLM stack, raising a clear error when it is missing."""
    try:
        import torch
        import torch.nn.functional as torch_functional
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except Exception as exc:  # pragma: no cover - exercised in lean installs
        raise RuntimeError(
            "LLM model loading requires the optional `llm` dependencies "
            "(`uv sync --extra llm`)."
        ) from exc
    return LLMDeps(
        torch,
        torch_functional,
        PeftModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )


def resolve_cuda_device(
    torch: Any, device: Any, *, label: str = "LLM"
) -> tuple[str, Optional[int]]:
    """Resolve a requested device to ``(device_str, cuda_index)``.

    Validates the requested GPU index (accepting ``int`` or ``"auto"``), selects
    it via ``set_device``, and falls back to GPU 0 with a warning on anything
    invalid. Returns ``("cpu", None)`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        logger.warning("%s: CUDA not available; using CPU.", label)
        return "cpu", None

    cuda_count = torch.cuda.device_count()
    if device == "auto" or device is None:
        idx = 0
    elif isinstance(device, bool):  # bool is an int subclass; reject it explicitly
        logger.warning("%s: invalid device %r; using 0.", label, device)
        idx = 0
    elif isinstance(device, int):
        idx = device
    else:
        try:
            idx = int(device)
        except (TypeError, ValueError):
            logger.warning("%s: non-numeric device %r; using 0.", label, device)
            idx = 0

    if idx < 0 or idx >= cuda_count:
        logger.warning(
            "%s: requested CUDA device %r invalid; using 0 (num_gpus=%d).",
            label,
            device,
            cuda_count,
        )
        idx = 0

    try:
        torch.cuda.set_device(idx)
    except Exception as e:
        logger.warning(
            "%s: torch.cuda.set_device(%d) failed: %s; using 0.", label, idx, e
        )
        idx = 0
        torch.cuda.set_device(idx)
    return f"cuda:{idx}", idx


def select_compute_dtype(torch: Any, use_cuda: bool) -> Any:
    """bfloat16 when supported, else float16 on GPU, else float32 on CPU."""
    if use_cuda and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if use_cuda:
        return torch.float16
    return torch.float32


def select_attn_impl(torch: Any, cuda_index: Optional[int]) -> Optional[str]:
    """Prefer FlashAttention-2 when installed and supported, else SDPA.

    Returns ``None`` on CPU (let transformers pick its default).
    """
    if cuda_index is None:
        return None
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401

        major, minor = torch.cuda.get_device_capability(cuda_index)
        if (major * 10 + minor) >= 75:
            attn_impl = "flash_attention_2"
            logger.info("Using FlashAttention-2.")
        else:
            logger.info("FlashAttention-2 unsupported on this GPU; using SDPA.")
    except Exception:
        logger.info("flash-attn not available; using SDPA.")
    return attn_impl


def build_4bit_quant_config(
    bnb_config: Any,
    *,
    load_in_4bit: bool,
    double_quant: bool = True,
    quant_type: str = "nf4",
    compute_dtype: Any = None,
) -> Any:
    """Build a BitsAndBytesConfig; a no-op config when ``load_in_4bit`` is False."""
    if not load_in_4bit:
        return bnb_config(load_in_4bit=False)
    return bnb_config(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def configure_decoder_tokenizer(tokenizer: Any) -> Any:
    """Left-pad/-truncate a decoder-only tokenizer and ensure a pad token exists.

    Decoder-only next-token prediction reads ``logits[:, -1, :]``. With right
    padding, that position is a PAD token for the shorter rows in a batch,
    producing wrong probabilities — so left padding is required for correctness.
    """
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
