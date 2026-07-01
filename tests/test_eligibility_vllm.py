import inspect

from trialmatchai.matching.eligibility_reasoning_vllm import BatchTrialProcessorVLLM


def test_vllm_processor_exposes_no_think():
    # rag.no_think must be able to reach the vLLM backend. It silently could not before:
    # the param was missing, so the inherited BaseTrialProcessor default (False) always won.
    params = inspect.signature(BatchTrialProcessorVLLM.__init__).parameters
    assert "no_think" in params
    assert params["no_think"].default is False


class _WhitespaceTokenizer:
    """Minimal tokenizer stub: one token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_vllm_trims_long_criteria_to_fit_context_window():
    # A long-criteria prompt must be trimmed (criteria only) so the model keeps room to finish
    # its CoT/JSON instead of overflowing the window and being dropped as invalid JSON.
    proc = BatchTrialProcessorVLLM.__new__(BatchTrialProcessorVLLM)  # bypass __init__ (no vLLM)
    proc.use_cot = False
    proc.no_think = False
    proc.llm = None
    proc.length_bucket = True
    proc.tokenizer = _WhitespaceTokenizer()

    proc._max_prompt_tokens = None
    overhead = proc._token_length(proc._format_prompt("", "patient description"))
    proc._max_prompt_tokens = overhead + 60

    long_criteria = " ".join(f"crit{i}" for i in range(2000))
    trimmed = proc._format_prompt(long_criteria, "patient description")
    assert proc._token_length(trimmed) <= proc._max_prompt_tokens  # now fits the window
    assert proc._token_length(trimmed) > overhead                  # but not all criteria dropped

    short = proc._format_prompt("Age >= 18 years", "patient description")
    assert "Age >= 18 years" in short  # a short trial is left untouched


def test_vllm_no_context_window_disables_trimming():
    proc = BatchTrialProcessorVLLM.__new__(BatchTrialProcessorVLLM)
    proc.use_cot = False
    proc.no_think = False
    proc.llm = None
    proc.length_bucket = True
    proc.tokenizer = _WhitespaceTokenizer()
    proc._max_prompt_tokens = None  # unknown window -> no trimming, prior behavior preserved

    long_criteria = " ".join(f"crit{i}" for i in range(5000))
    prompt = proc._format_prompt(long_criteria, "patient")
    assert "crit4999" in prompt  # last criterion survives — nothing was trimmed
