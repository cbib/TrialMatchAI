"""Reranker builds through the shared vLLM loader with knob-parity (Phase 5, RC4)."""

import sys
import types

import pytest


@pytest.fixture
def stub_vllm(monkeypatch):
    vllm_stub = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    vllm_stub.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)

    import trialmatchai.models.llm.vllm_loader as vl

    captured = {}

    class FakeTok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [1 if s == "Yes" else 2]}

        def apply_chat_template(self, m, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

    def fake_loader(model_config, vllm_cfg):
        captured["model_config"] = model_config
        captured["vllm_cfg"] = vllm_cfg
        return ("ENGINE", FakeTok(), "LORA")

    monkeypatch.setattr(vl, "load_vllm_engine", fake_loader)
    return captured


def test_reranker_uses_shared_loader_with_knob_parity(stub_vllm):
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    r = LLMReranker(
        model_path="google/gemma-2-2b-it",
        adapter_path="models/finetuned_gemma2",
        gpu_memory_utilization=0.15,
        max_model_len=4096,
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        batch_size=20,
    )
    mc, vc = stub_vllm["model_config"], stub_vllm["vllm_cfg"]
    assert mc["base_model"] == "google/gemma-2-2b-it"
    assert mc["cot_adapter_path"] == "models/finetuned_gemma2"
    # Knobs the inline engine could never take now reach the reranker (RC4 parity):
    assert vc["quantization"] == "bitsandbytes"
    assert vc["kv_cache_dtype"] == "fp8"
    assert vc["enforce_eager"] is True
    assert vc["adapter_name"] == "reranker_adapter"
    assert r.llm == "ENGINE" and r.lora_request == "LORA"


def test_reranker_yes_no_scoring_unchanged(stub_vllm):
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    r = LLMReranker(model_path="m", adapter_path=None)

    class LP:
        def __init__(self, v):
            self.logprob = v

    out = types.SimpleNamespace(
        outputs=[types.SimpleNamespace(logprobs=[{1: LP(0.0), 2: LP(-2.0)}])]
    )
    # softmax(0, -2) over Yes/No ~= 0.88
    assert 0.87 < r._yes_probability(out) < 0.89
