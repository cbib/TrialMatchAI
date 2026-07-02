"""Offline evaluation and threshold tuning for the concept-linker accept gate.

Thresholds are tuned on a labeled dev set with a NIL-aware metric: standard EL benchmarks
score only in-KB gold mentions and never penalize always-linking, so NIL is a first-class
class here and the tuning objective is macro-F1 over {correct-link, correct-NIL}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trialmatchai.entities.linker import _lexical_score, gate_status, lexical_reranker
from trialmatchai.entities.types import NO_ENTITY_ID

_METRIC_KEYS = (
    "accuracy",
    "link_precision",
    "link_recall",
    "link_f1",
    "nil_precision",
    "nil_recall",
    "nil_f1",
    "macro_f1",
)


@dataclass(frozen=True)
class LinkExample:
    """A labeled mention. ``gold_id`` is the normalized concept id, or None for a NIL mention."""

    mention: str
    entity_group: str
    gold_id: str | None


@dataclass(frozen=True)
class GateInput:
    """Cached retrieval for one mention: gold id + (normalized_id, lexical_quality) per
    candidate, ordered best-first (after reranking). Lets a threshold sweep avoid re-querying."""

    gold_id: str | None
    ranked: tuple[tuple[str, float], ...]


def _is_nil(value: str | None) -> bool:
    return value is None or value == NO_ENTITY_ID


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def linking_metrics(
    gold: Sequence[str | None], predicted: Sequence[str | None]
) -> dict[str, float]:
    """NIL-aware linking metrics; None or NO_ENTITY_ID counts as an abstention (NIL).

    link_* is scored over mentions the system links / that have a gold concept; nil_* over the
    NIL class; macro_f1 is the mean of link_f1 and nil_f1 (the tuning objective).
    """
    n = len(gold)
    if n == 0 or n != len(predicted):
        return {key: 0.0 for key in _METRIC_KEYS}
    pairs = list(zip(gold, predicted))
    correct = sum(
        1
        for g, p in pairs
        if (_is_nil(g) and _is_nil(p)) or (not _is_nil(g) and g == p)
    )
    tp_link = sum(1 for g, p in pairs if not _is_nil(g) and g == p)
    pred_links = sum(1 for p in predicted if not _is_nil(p))
    gold_links = sum(1 for g in gold if not _is_nil(g))
    tp_nil = sum(1 for g, p in pairs if _is_nil(g) and _is_nil(p))
    pred_nil = sum(1 for p in predicted if _is_nil(p))
    gold_nil = sum(1 for g in gold if _is_nil(g))

    link_p = tp_link / pred_links if pred_links else 0.0
    link_r = tp_link / gold_links if gold_links else 0.0
    nil_p = tp_nil / pred_nil if pred_nil else 0.0
    nil_r = tp_nil / gold_nil if gold_nil else 0.0
    link_f1 = _f1(link_p, link_r)
    nil_f1 = _f1(nil_p, nil_r)
    return {
        "accuracy": correct / n,
        "link_precision": link_p,
        "link_recall": link_r,
        "link_f1": link_f1,
        "nil_precision": nil_p,
        "nil_recall": nil_r,
        "nil_f1": nil_f1,
        "macro_f1": (link_f1 + nil_f1) / 2,
    }


def predict(input_row: GateInput, *, accept: float, reject: float, margin: float) -> str | None:
    """Apply the accept gate to one cached mention; returns the linked id or None (abstain)."""
    if not input_row.ranked:
        return None
    top_id, quality = input_row.ranked[0]
    runner_up = input_row.ranked[1][1] if len(input_row.ranked) > 1 else 0.0
    status = gate_status(
        quality,
        runner_up=runner_up,
        accept_threshold=accept,
        reject_threshold=reject,
        margin=margin,
    )
    return top_id if status == "accepted" else None


def build_gate_inputs(
    linker, examples: Sequence[LinkExample], *, rerank: bool = True
) -> list[GateInput]:
    """Run retrieval once per example and cache (normalized_id, lexical_quality) per candidate."""
    inputs: list[GateInput] = []
    for example in examples:
        schema = linker.schemas_by_label.get(example.entity_group.casefold())
        if schema is None or not schema.is_linkable or linker.store is None:
            inputs.append(GateInput(example.gold_id, ()))
            continue
        candidates = list(
            linker.store.search(
                example.mention,
                vocabularies=schema.target_vocabularies,
                domain_hints=schema.domain_hints,
                limit=linker.search_limit,
            )
        )
        if rerank:
            candidates = list(lexical_reranker(example.mention, candidates))
        ranked = tuple(
            (candidate.normalized_id, _lexical_score(example.mention, candidate))
            for candidate in candidates
        )
        inputs.append(GateInput(example.gold_id, ranked))
    return inputs


def sweep_thresholds(
    inputs: Sequence[GateInput],
    accept_grid: Sequence[float],
    *,
    reject: float,
    margin: float,
) -> list[dict[str, float]]:
    """Evaluate each accept threshold in the grid; returns one metrics row per threshold."""
    gold = [row.gold_id for row in inputs]
    rows: list[dict[str, float]] = []
    for accept in accept_grid:
        preds = [predict(row, accept=accept, reject=reject, margin=margin) for row in inputs]
        rows.append(
            {
                "accept_threshold": accept,
                "reject_threshold": reject,
                "margin": margin,
                **linking_metrics(gold, preds),
            }
        )
    return rows


def best_accept_threshold(
    inputs: Sequence[GateInput],
    accept_grid: Sequence[float],
    *,
    reject: float,
    margin: float,
) -> dict[str, float]:
    """Return the sweep row with the highest macro-F1 (ties: the lowest threshold)."""
    rows = sweep_thresholds(inputs, accept_grid, reject=reject, margin=margin)
    return max(rows, key=lambda row: row["macro_f1"])
