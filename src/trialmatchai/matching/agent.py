"""A bounded per-patient controller over retrieval width and shortlist depth.

The pipeline is otherwise a single pass with hardcoded widths: the second level retrieves
``size=250`` criteria per query and the shortlist takes a fixed slice of whatever that
produced. Measurement says both numbers are wrong, and wrong in a patient-dependent way:

  * The depth a patient needs to reach 90% of its own first-level recall ranges from 50 to
    1550 trials, spread evenly across that range (17 of 75 TREC-2021 patients need <=200,
    21 need >700). One number cannot serve both ends; sizing for the worst case wastes ~65%
    of the reasoner's compute and sizing for the median drops the hard patients.
  * The second level surfaces ~446 unique trials from a ~1860-trial candidate pool, and the
    shortlist can only choose among those. Every downstream knob sits behind that.

So the agent's job is ALLOCATION, not extra reasoning. Three attempts to add a reasoning
component to this pipeline measured neutral or worse (constraint verification, uncertainty
gating, softened disqualification), while changing how compute is distributed is the one
intervention that has held up. This module does only that.

Design constraints, taken from the failure literature rather than invented:

  * **Closed action set.** Four actions, no free-form tool choice. The MAST study of 1,600+
    traces of popular multi-agent frameworks found 41-87% task failure, dominated by
    specification (41.8%) and coordination (36.9%) problems -- i.e. system design, not model
    quality. A small fixed action set is what avoids that class of failure.
  * **Hard budget.** Rounds and total criteria examined are both capped. The agent cannot
    spend without bound however promising things look.
  * **Monotone widening.** Each round only ever ADDS candidates. No round removes something
    an earlier round accepted, so errors cannot compound across rounds.
  * **No LLM in the control loop.** Every decision is a pure function of counts and scores
    the pipeline already computes. The agent is auditable, reproducible, and free.

Disabled by default (``agent.enabled``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

Action = Literal["widen_retrieval", "deepen_shortlist", "stop"]

# Defaults chosen from the measured curves, not tuned. per_query_size starts at the current
# hardcoded 250 so round 1 reproduces today's behaviour exactly; widening is what the agent
# adds on top.
DEFAULT_MAX_ROUNDS = 3
DEFAULT_START_SIZE = 250
DEFAULT_SIZE_MULTIPLIER = 2.0
DEFAULT_MAX_SIZE = 2000
# Stop widening when a round's extra retrieval buys less than this fraction of new trials
# relative to what was already surfaced. At 0.10 a round must grow the pool by >=10% to
# justify the next one.
DEFAULT_MIN_YIELD = 0.10
# Never let one patient consume more than this many criteria across all rounds.
DEFAULT_MAX_CRITERIA = 40_000


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_criteria: int = DEFAULT_MAX_CRITERIA
    max_size: int = DEFAULT_MAX_SIZE

    def exhausted(self, *, rounds_done: int, criteria_examined: int) -> bool:
        return rounds_done >= self.max_rounds or criteria_examined >= self.max_criteria


@dataclass
class RoundObservation:
    """What the agent can see after one second-level round. Counts only -- no model calls."""

    round_index: int
    per_query_size: int
    criteria_examined: int
    trials_surfaced: int
    new_trials: int
    candidate_pool: int

    @property
    def yield_rate(self) -> float:
        """New trials this round as a fraction of what was already surfaced."""
        previously = self.trials_surfaced - self.new_trials
        if previously <= 0:
            return 1.0 if self.new_trials else 0.0
        return self.new_trials / previously

    @property
    def pool_coverage(self) -> float:
        if self.candidate_pool <= 0:
            return 1.0
        return self.trials_surfaced / self.candidate_pool


@dataclass
class AgentTrace:
    """Provenance for the whole episode, written beside the shortlist."""

    rounds: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    stopped_because: str = ""

    def record(self, observation: RoundObservation, action: Action, reason: str) -> None:
        self.rounds.append(
            {
                "round": observation.round_index,
                "per_query_size": observation.per_query_size,
                "criteria_examined": observation.criteria_examined,
                "trials_surfaced": observation.trials_surfaced,
                "new_trials": observation.new_trials,
                "yield_rate": round(observation.yield_rate, 4),
                "pool_coverage": round(observation.pool_coverage, 4),
                "action": action,
                "reason": reason,
            }
        )
        self.actions.append(action)
        if action == "stop":
            self.stopped_because = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "actions": self.actions,
            "stopped_because": self.stopped_because,
            "total_criteria_examined": sum(r["criteria_examined"] for r in self.rounds),
            "final_trials_surfaced": self.rounds[-1]["trials_surfaced"] if self.rounds else 0,
        }


def agent_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (config or {}).get("agent") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_rounds": int(raw.get("max_rounds", DEFAULT_MAX_ROUNDS)),
        "start_size": int(raw.get("start_size", DEFAULT_START_SIZE)),
        "size_multiplier": float(raw.get("size_multiplier", DEFAULT_SIZE_MULTIPLIER)),
        "max_size": int(raw.get("max_size", DEFAULT_MAX_SIZE)),
        "min_yield": float(raw.get("min_yield", DEFAULT_MIN_YIELD)),
        "max_criteria": int(raw.get("max_criteria", DEFAULT_MAX_CRITERIA)),
    }


def budget_from_config(config: Mapping[str, Any] | None) -> AgentBudget:
    cfg = agent_config(config)
    return AgentBudget(
        max_rounds=cfg["max_rounds"],
        max_criteria=cfg["max_criteria"],
        max_size=cfg["max_size"],
    )


def decide(
    observation: RoundObservation,
    budget: AgentBudget,
    *,
    min_yield: float = DEFAULT_MIN_YIELD,
    criteria_examined: int = 0,
) -> tuple[Action, str]:
    """The whole policy. A pure function of counts -- no model, no hidden state.

    Widening is justified only while it is still finding trials the previous rounds missed.
    ``yield_rate`` is measured against what was already surfaced rather than against the
    candidate pool, because the pool is a poor denominator: a patient with 2000 candidates
    and 400 genuinely plausible trials should stop at 400, not chase 20% coverage.
    """
    rounds_done = observation.round_index + 1
    if budget.exhausted(rounds_done=rounds_done, criteria_examined=criteria_examined):
        return "stop", (
            f"budget exhausted (rounds {rounds_done}/{budget.max_rounds}, "
            f"criteria {criteria_examined}/{budget.max_criteria})"
        )
    if observation.trials_surfaced >= observation.candidate_pool > 0:
        return "stop", "whole candidate pool already surfaced"
    if observation.per_query_size >= budget.max_size:
        return "deepen_shortlist", f"per-query size at the cap ({budget.max_size})"
    if observation.yield_rate < min_yield:
        return "stop", (
            f"marginal yield {observation.yield_rate:.3f} below {min_yield:.3f}; "
            "widening is no longer finding new trials"
        )
    return "widen_retrieval", (
        f"yield {observation.yield_rate:.3f} >= {min_yield:.3f} and budget remains"
    )


def next_size(current: int, *, multiplier: float, cap: int) -> int:
    return max(current + 1, min(cap, int(round(current * multiplier))))


def plan_sizes(config: Mapping[str, Any] | None) -> Sequence[int]:
    """The size ladder the agent would climb if every round justified widening.

    Exposed so a run can be costed before it is launched.
    """
    cfg = agent_config(config)
    sizes, size = [], cfg["start_size"]
    for _ in range(max(1, cfg["max_rounds"])):
        sizes.append(size)
        if size >= cfg["max_size"]:
            break
        size = next_size(size, multiplier=cfg["size_multiplier"], cap=cfg["max_size"])
    return sizes
