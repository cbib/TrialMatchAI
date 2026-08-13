"""Reranker scoring modes (models/llm/llm_reranker.py).

The scores this produces are SUMMED and AVERAGED into a trial score and then cut at 0.5
(criteria_retrieval.aggregate_to_trials), so what matters is not just the ordering but the
magnitude. These tests pin the [0, 1] range, the binary contract, and the graded spread.
"""

import math
import types

import pytest

from trialmatchai.models.llm.llm_reranker import GRADED_LABELS, LLMReranker


def _reranker(scoring, label_token_ids):
    """An LLMReranker without __init__ -- no vLLM engine, no GPU."""
    r = LLMReranker.__new__(LLMReranker)
    r.scoring = scoring
    r.label_token_ids = label_token_ids
    r.applicable_token_id, r.not_applicable_token_id = 1, 0
    return r


def _output(logprob_by_token):
    entries = {t: types.SimpleNamespace(logprob=lp) for t, lp in logprob_by_token.items()}
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[entries])])


# --------------------------------------------------------------- binary contract
def test_binary_returns_probability_of_yes():
    r = _reranker("binary", [0, 1])  # [No, Yes]
    # Equal logprobs -> 0.5 exactly.
    assert r._yes_probability(_output({0: math.log(0.5), 1: math.log(0.5)})) == pytest.approx(0.5)
    # Yes dominant.
    assert r._yes_probability(_output({0: math.log(0.1), 1: math.log(0.9)})) == pytest.approx(0.9)
    # No dominant.
    assert r._yes_probability(_output({0: math.log(0.8), 1: math.log(0.2)})) == pytest.approx(0.2)


def test_binary_matches_the_historical_softmax_formula():
    """Regression against the original two-token implementation."""
    yes_lp, no_lp = -0.3, -1.7
    highest = max(yes_lp, no_lp)
    expected = math.exp(yes_lp - highest) / (
        math.exp(yes_lp - highest) + math.exp(no_lp - highest)
    )
    r = _reranker("binary", [0, 1])
    assert r._yes_probability(_output({0: no_lp, 1: yes_lp})) == pytest.approx(expected)


# --------------------------------------------------------------- graded contract
def test_graded_spans_the_unit_interval():
    r = _reranker("graded", [10, 11, 12])  # Not, Somewhat, Highly
    big, small = math.log(0.999), math.log(0.0005)
    assert r._yes_probability(_output({10: big, 11: small, 12: small})) == pytest.approx(0.0, abs=1e-3)
    assert r._yes_probability(_output({10: small, 11: big, 12: small})) == pytest.approx(0.5, abs=1e-3)
    assert r._yes_probability(_output({10: small, 11: small, 12: big})) == pytest.approx(1.0, abs=1e-3)


def test_graded_produces_intermediate_values_where_binary_saturates():
    """The point of the change. Binary pushes mass to the extremes; the graded expectation
    lands in the partial-relevance band that aggregation and the 0.5 cut depend on."""
    r = _reranker("graded", [10, 11, 12])
    score = r._yes_probability(
        _output({10: math.log(0.25), 11: math.log(0.5), 12: math.log(0.25)})
    )
    assert score == pytest.approx(0.5, abs=1e-6)
    assert 0.0 < score < 1.0

    skewed = r._yes_probability(
        _output({10: math.log(0.1), 11: math.log(0.3), 12: math.log(0.6)})
    )
    assert skewed == pytest.approx((0 * 0.1 + 1 * 0.3 + 2 * 0.6) / 2)
    assert 0.5 < skewed < 1.0


def test_graded_is_monotonic_in_the_top_label():
    r = _reranker("graded", [10, 11, 12])
    scores = [
        r._yes_probability(_output({10: math.log(1 - p - 0.05), 11: math.log(0.05), 12: math.log(p)}))
        for p in (0.1, 0.3, 0.5, 0.7)
    ]
    assert scores == sorted(scores)


@pytest.mark.parametrize("mode,ids", [("binary", [0, 1]), ("graded", [10, 11, 12])])
def test_scores_stay_in_unit_interval(mode, ids):
    r = _reranker(mode, ids)
    for lp in (-0.01, -1.0, -12.0):
        score = r._yes_probability(_output({t: lp * (i + 1) for i, t in enumerate(ids)}))
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------- robustness
@pytest.mark.parametrize("mode,ids", [("binary", [0, 1]), ("graded", [10, 11, 12])])
def test_missing_label_tokens_do_not_raise(mode, ids):
    """vLLM returns the top-k logprobs; a label may be absent. Renormalizing over what IS
    present must still yield a usable score rather than an exception."""
    r = _reranker(mode, ids)
    score = r._yes_probability(_output({ids[-1]: math.log(0.9)}))
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("mode,ids", [("binary", [0, 1]), ("graded", [10, 11, 12])])
def test_malformed_output_scores_zero(mode, ids):
    r = _reranker(mode, ids)
    assert r._yes_probability(types.SimpleNamespace(outputs=[])) == 0.0
    assert r._yes_probability(_output({999: math.log(0.5)})) == 0.0


# --------------------------------------------------------------- prompts
def test_binary_prompt_is_unchanged():
    """The LoRA adapter was tuned against this exact wording; changing it silently would
    invalidate the adapter."""
    messages = LLMReranker.create_messages("patient", "criterion", scoring="binary")
    assert "sufficient information" in messages[0]["content"]
    assert messages[0]["content"] == LLMReranker.BINARY_SYSTEM_PROMPT


def test_graded_prompt_asks_for_relevance_not_answerability():
    """The substantive fix: the binary prompt asks whether the criterion CAN BE EVALUATED,
    which is answerability, not relevance."""
    messages = LLMReranker.create_messages("patient", "criterion", scoring="graded")
    prompt = messages[0]["content"]
    assert "relevant" in prompt.lower()
    assert "sufficient information" not in prompt
    for label in GRADED_LABELS:
        assert label in prompt


def test_both_prompts_carry_the_same_statement_payload():
    for mode in ("binary", "graded"):
        messages = LLMReranker.create_messages("PT", "CR", scoring=mode)
        assert messages[-1]["content"] == "Statement A: PT\nStatement B: CR\n\n"
