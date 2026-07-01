import inspect

from trialmatchai.matching.eligibility_reasoning_vllm import BatchTrialProcessorVLLM


def test_vllm_processor_exposes_no_think():
    # rag.no_think must be able to reach the vLLM backend. It silently could not before:
    # the param was missing, so the inherited BaseTrialProcessor default (False) always won.
    params = inspect.signature(BatchTrialProcessorVLLM.__init__).parameters
    assert "no_think" in params
    assert params["no_think"].default is False
