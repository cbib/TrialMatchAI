"""Unit tests for the shared LLM helpers (models/llm/_common.py).

These lock the correctness fixes from PR2 — left padding, device resolution,
dtype selection — using lightweight stubs so they run without the `llm` extra.
"""

from trialmatchai.models.llm._common import (
    configure_decoder_tokenizer,
    resolve_cuda_device,
    select_attn_impl,
    select_compute_dtype,
)


class FakeTokenizer:
    def __init__(self, pad_token=None, eos_token="</s>"):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = "right"
        self.truncation_side = "right"


class FakeCuda:
    def __init__(self, available=True, count=2, bf16=False):
        self._available = available
        self._count = count
        self._bf16 = bf16
        self.selected = None

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def is_bf16_supported(self):
        return self._bf16

    def set_device(self, idx):
        self.selected = idx


class FakeTorch:
    def __init__(self, **kw):
        self.cuda = FakeCuda(**kw)
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"


def test_configure_decoder_tokenizer_sets_left_padding_and_pad_token():
    tok = FakeTokenizer(pad_token=None, eos_token="<eos>")
    configure_decoder_tokenizer(tok)
    assert tok.padding_side == "left"
    assert tok.truncation_side == "left"
    assert tok.pad_token == "<eos>"  # filled from eos when missing


def test_configure_decoder_tokenizer_keeps_existing_pad_token():
    tok = FakeTokenizer(pad_token="<pad>", eos_token="<eos>")
    configure_decoder_tokenizer(tok)
    assert tok.pad_token == "<pad>"


def test_resolve_cuda_device_cpu_when_unavailable():
    torch = FakeTorch(available=False)
    assert resolve_cuda_device(torch, 0) == ("cpu", None)


def test_resolve_cuda_device_auto_selects_zero():
    torch = FakeTorch(available=True, count=2)
    assert resolve_cuda_device(torch, "auto") == ("cuda:0", 0)
    assert torch.cuda.selected == 0


def test_resolve_cuda_device_valid_index():
    torch = FakeTorch(available=True, count=2)
    assert resolve_cuda_device(torch, 1) == ("cuda:1", 1)
    assert torch.cuda.selected == 1


def test_resolve_cuda_device_invalid_index_falls_back_to_zero():
    torch = FakeTorch(available=True, count=2)
    assert resolve_cuda_device(torch, 5) == ("cuda:0", 0)


def test_resolve_cuda_device_non_numeric_string_falls_back():
    torch = FakeTorch(available=True, count=2)
    assert resolve_cuda_device(torch, "gpu0") == ("cuda:0", 0)


def test_select_compute_dtype():
    assert select_compute_dtype(FakeTorch(available=True, bf16=True), True) == "bfloat16"
    assert select_compute_dtype(FakeTorch(available=True, bf16=False), True) == "float16"
    assert select_compute_dtype(FakeTorch(available=False), False) == "float32"


def test_select_attn_impl_cpu_is_none():
    assert select_attn_impl(FakeTorch(available=False), None) is None


def test_select_attn_impl_gpu_without_flash_attn_is_sdpa():
    # flash_attn is not installed in the base test env, so this falls back.
    assert select_attn_impl(FakeTorch(available=True), 0) == "sdpa"
