"""Second-level width and adaptive output budget.

Two hardcoded constants were the pipeline's binding bottleneck: the per-query criteria
retrieval size (250) and the aggregation threshold (0.5). Measured on TREC 2023, size alone
capped the second level at 43% of its candidate pool. These tests pin the new config surface
and, importantly, that the defaults reproduce the old behaviour.
"""

import pytest

from trialmatchai.matching.eligibility_base import count_criteria
from trialmatchai.matching.eligibility_reasoning_vllm import adaptive_max_tokens
from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever


def _retriever(**kwargs):
    return SecondStageRetriever(
        search_backend=object(), llm_reranker=None, embedder=None, **kwargs
    )


# ----------------------------------------------------------------- width defaults
def test_defaults_reproduce_the_previous_hardcoded_behaviour():
    r = _retriever()
    assert r.size == 250
    assert r.aggregation_threshold == 0.5
    assert r.aggregation_method == "weighted"


def test_width_is_configurable():
    r = _retriever(size=1000, aggregation_threshold=0.2, aggregation_method="sqrt")
    assert r.size == 1000
    assert r.aggregation_threshold == 0.2
    assert r.aggregation_method == "sqrt"


def _criterion(nct, score, cid):
    return {"_source": {"nct_id": nct, "criteria_id": cid}, "llm_score": score}


def test_aggregation_threshold_gates_which_trials_survive():
    """The 0.5 cut drops a trial whose every criterion scores below it -- that trial then
    never reaches the shortlist, whatever the reranker thought."""
    criteria = [_criterion("NCT1", 0.9, "a"), _criterion("NCT2", 0.3, "b")]

    strict = _retriever().aggregate_to_trials(criteria)
    assert {t["nct_id"] for t in strict} == {"NCT1"}  # NCT2 dropped at 0.5

    lenient = _retriever(aggregation_threshold=0.2).aggregate_to_trials(criteria)
    assert {t["nct_id"] for t in lenient} == {"NCT1", "NCT2"}


def test_explicit_argument_still_overrides_the_configured_threshold():
    criteria = [_criterion("NCT2", 0.3, "b")]
    r = _retriever(aggregation_threshold=0.5)
    assert r.aggregate_to_trials(criteria) == []
    assert len(r.aggregate_to_trials(criteria, threshold=0.1)) == 1


# ----------------------------------------------------------------- criterion counting
def test_count_criteria_ignores_headers_and_bullets():
    text = (
        "Inclusion Criteria:\n"
        "- Age 18 or older\n"
        "- Histologically confirmed glioma\n"
        "\n"
        "Exclusion Criteria:\n"
        "* Prior systemic therapy\n"
    )
    assert count_criteria(text) == 3


@pytest.mark.parametrize("value", ["", None, "   \n  \n"])
def test_count_criteria_handles_empty(value):
    assert count_criteria(value) == 0


def test_count_criteria_accepts_a_list():
    assert count_criteria(["a", "b", ""]) == 2


# ----------------------------------------------------------------- adaptive budget
def test_budget_scales_with_criterion_count():
    small = adaptive_max_tokens(5, ceiling=8192)
    typical = adaptive_max_tokens(17, ceiling=8192)
    large = adaptive_max_tokens(60, ceiling=8192)
    assert small < typical < large


def test_typical_trial_lands_near_the_measured_medical_optimum():
    """m1 puts the medical reasoning optimum near 4K tokens. The TREC 2023 mean trial carries
    ~17 criteria, so that case should land in the same neighbourhood -- not at an 8K default."""
    assert 3000 <= adaptive_max_tokens(17, ceiling=8192) <= 5000


def test_budget_never_exceeds_the_ceiling():
    """The whole safety argument: this can only LOWER the budget, so no trial is truncated
    more than it already would be under the fixed setting."""
    for n in (0, 1, 17, 60, 500):
        assert adaptive_max_tokens(n, ceiling=8192) <= 8192
        assert adaptive_max_tokens(n, ceiling=2048) <= 2048


def test_large_trials_keep_full_headroom():
    """A 60-criterion trial must still get the whole configured budget rather than a 4K cap
    that would truncate it into invalid JSON."""
    assert adaptive_max_tokens(60, ceiling=8192) == 8192


def test_budget_has_a_floor_for_degenerate_input():
    assert adaptive_max_tokens(0, ceiling=8192) >= 1024
    # A tiny ceiling still wins -- the floor never pushes past what the caller allows.
    assert adaptive_max_tokens(0, ceiling=256) == 256
