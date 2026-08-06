#!/usr/bin/env python
"""Compare the arms of the shortlist-policy A/B (scripts/run_trec_shortlist_ab.slurm).

Reports each arm against the fixed baseline, and -- crucially -- separates the two
effects that a depth policy produces:

  * spending compute BETTER  (equal-cost arm: same mean shortlist size, more recall)
  * spending compute MORE    (deeper arm: bigger shortlist, bought with GPU time)

Conflating them overstates the result, so cost is always printed beside quality.

    uv run python scripts/compare_shortlist_ab.py [--root shortlist_ab] [--track 21]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE = "fixed"
QUALITY = (
    "shortlist_recall",
    "recall@1000",
    "ndcg@10",
    "P@10(rel>=1)",
    "P@10(eligible)",
)
COST = ("shortlist_size",)
DIAGNOSTIC = ("funnel_depth_loss", "shortlist_selection_delta")


def load_arm(root: Path, arm: str, track: str) -> dict | None:
    path = root / arm / f"results_trec{track}" / "evaluation_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {"mean": data.get("mean", {}), "n": data.get("num_queries_scored")}


def fmt(value: object, width: int = 9) -> str:
    if isinstance(value, (int, float)):
        return f"{value:{width}.4f}"
    return f"{'--':>{width}}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="shortlist_ab")
    parser.add_argument("--track", default="21")
    args = parser.parse_args()

    root = Path(args.root)
    arms = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if BASELINE not in arms:
        print(f"No '{BASELINE}' arm under {root}/ -- nothing to compare against.")
        return 1

    loaded = {arm: load_arm(root, arm, args.track) for arm in arms}
    missing = [arm for arm, data in loaded.items() if data is None]
    for arm in missing:
        print(f"NOTE: arm '{arm}' has no evaluation_metrics.json yet (still running?)")
        loaded.pop(arm)
    if BASELINE not in loaded:
        return 1

    base = loaded[BASELINE]["mean"]
    order = [BASELINE] + [a for a in loaded if a != BASELINE]

    print(f"\nTREC {args.track} shortlist-policy A/B  (n={loaded[BASELINE]['n']} topics)\n")
    label_w = max(len(a) for a in order) + 2
    for group, keys in (("COST", COST), ("QUALITY", QUALITY), ("DIAGNOSTIC", DIAGNOSTIC)):
        print(f"  {group}")
        print(f"    {'arm':<{label_w}}" + "".join(f"{k:>26s}" for k in keys))
        for arm in order:
            mean = loaded[arm]["mean"]
            cells = ""
            for key in keys:
                value = mean.get(key)
                cell = fmt(value)
                if arm != BASELINE and isinstance(value, (int, float)):
                    ref = base.get(key)
                    if isinstance(ref, (int, float)):
                        cell += f" ({value - ref:+.4f})"
                cells += f"{cell:>26s}"
            print(f"    {arm:<{label_w}}" + cells)
        print()

    print("  Reading this:")
    print("    An arm at the SAME shortlist_size as 'fixed' with higher shortlist_recall")
    print("    spends its compute better -- that gain is free.")
    print("    An arm with a LARGER shortlist_size bought its gain with GPU time; compare")
    print("    it against 'fixed' only after noting the extra cost.")
    print("    shortlist_selection_delta < 0 means the second level is still selecting worse")
    print("    than a plain first-level cut at that depth, independent of the policy.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
