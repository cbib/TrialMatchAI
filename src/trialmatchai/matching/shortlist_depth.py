"""How many trials the eligibility reasoner gets to read.

The shortlist is the pipeline's narrowest point: a relevant trial dropped here can
never be ranked, however good the reasoning is. Measured on the completed TREC runs,
a fixed-size shortlist discards a third of the relevant trials the first level had
already found (``shortlist_recall`` in ``trec/qrels.py``).

Depth is the dominant cause -- 94% of that loss on TREC 2021 -- and no single number
serves every patient: the depth needed to reach 90% of a patient's own first-level
recall ranges from 50 to 1550 trials, spread evenly across that range. Sizing for the
worst case wastes ~65% of the reasoner's compute; sizing for the median silently drops
the hard patients.

``relative_to_max`` therefore reads the shape of the first-level score curve. A peaked
curve means retrieval was confident and few trials are plausible; a flat curve means
many are, and the patient needs more depth. Offline replay over the completed runs
(first-level scores, same mean depth as the fixed policy) gives:

    TREC 2021   +0.017 to +0.029 recall
    TREC 2022   +0.007 to +0.025 recall
    TREC 2023   -0.010 to +0.006 recall  (questionnaire topics; no gain)

So it is a small, free gain on the narrative-topic tracks and a wash on 2023 -- worth
having, but not a substitute for spending more depth outright. ``fixed`` remains the
default so enabling this is an explicit A/B, not a silent change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

POLICIES = ("fixed", "relative_to_max")
DEFAULT_ALPHA = 0.25
DEFAULT_MIN_DEPTH = 50


def shortlist_config(search_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve the ``search.shortlist`` block, tolerating absent or partial config."""
    raw = (search_config or {}).get("shortlist") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    policy = str(raw.get("policy", "fixed") or "fixed")
    if policy not in POLICIES:
        logger.warning(
            "Unknown search.shortlist.policy %r; falling back to 'fixed'. Known: %s",
            policy,
            ", ".join(POLICIES),
        )
        policy = "fixed"
    return {
        "policy": policy,
        "relative_to_max_alpha": float(raw.get("relative_to_max_alpha", DEFAULT_ALPHA)),
        "min_depth": int(raw.get("min_depth", DEFAULT_MIN_DEPTH)),
        "max_depth": raw.get("max_depth"),
    }


def _relative_to_max_depth(scores: list[float], alpha: float) -> int:
    """Count of trials scoring at least ``alpha`` x the top score.

    Scores are a weighted RRF sum, so they are positive and comparable only within one
    patient -- which is exactly why the cut is relative to that patient's own maximum
    rather than an absolute threshold.
    """
    if not scores:
        return 0
    top = scores[0]
    if top <= 0:
        return len(scores)
    cut = alpha * top
    kept = 0
    for score in scores:
        if score < cut:
            break
        kept += 1
    return kept


def choose_shortlist_depth(
    *,
    first_level_scores: Mapping[str, float] | None,
    fixed_depth: int,
    upper_bound: int,
    search_config: Mapping[str, Any] | None = None,
) -> int:
    """Shortlist size for one patient.

    ``fixed_depth`` is what the existing divisor-based sizing would have chosen, and is
    returned unchanged under the default policy. ``upper_bound`` is the hard ceiling the
    caller can honour (the reasoner's own cap), and is never exceeded.
    """
    upper_bound = max(1, int(upper_bound))
    fixed_depth = max(1, min(int(fixed_depth), upper_bound))
    cfg = shortlist_config(search_config)
    if cfg["policy"] == "fixed":
        return fixed_depth

    scores = sorted((float(v) for v in (first_level_scores or {}).values()), reverse=True)
    if not scores:
        # No first-level signal to read (e.g. a resumed run missing the scores file):
        # degrade to the fixed sizing rather than guessing a depth.
        logger.warning(
            "shortlist policy 'relative_to_max' has no first-level scores; using fixed depth %s",
            fixed_depth,
        )
        return fixed_depth

    depth = _relative_to_max_depth(scores, cfg["relative_to_max_alpha"])
    floor = max(1, cfg["min_depth"])
    ceiling = upper_bound
    configured_max = cfg["max_depth"]
    if configured_max is not None:
        ceiling = min(ceiling, max(1, int(configured_max)))
    depth = max(floor, min(depth, ceiling))
    logger.info(
        "Shortlist depth %s (policy=relative_to_max, alpha=%.3g, fixed would be %s)",
        depth,
        cfg["relative_to_max_alpha"],
        fixed_depth,
    )
    return depth


def depth_report(
    *,
    chosen: int,
    fixed_depth: int,
    first_level_scores: Mapping[str, float] | None,
    search_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance for the depth decision, written beside the shortlist."""
    cfg = shortlist_config(search_config)
    scores: Iterable[float] = (first_level_scores or {}).values()
    ordered = sorted((float(v) for v in scores), reverse=True)
    return {
        "policy": cfg["policy"],
        "chosen_depth": int(chosen),
        "fixed_depth": int(fixed_depth),
        "relative_to_max_alpha": cfg["relative_to_max_alpha"],
        "min_depth": cfg["min_depth"],
        "max_depth": cfg["max_depth"],
        "candidate_pool": len(ordered),
        "top_score": ordered[0] if ordered else None,
    }
