#!/usr/bin/env python
"""Print the actual terms llm_expansion produces, for inspection.

The A/B measures whether the channel lifts recall; this shows WHAT it is searching for, which
is what tells you whether the terms are clinically sensible or whether the channel is quietly
adding noise at weight 0.5.

Prints, per patient: the matching summary the expander reads, the six fields it returns, and
the flattened term list the planner actually turns into a query channel (capped at
llm_max_terms and spent in field order, so the cap can starve the later fields).

    python scripts/show_llm_expansions.py --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARIES = Path("data/patients/trec23/summaries")
PROFILES = Path("data/patients/trec23/profiles")
BASE_CFG = "src/trialmatchai/config/config_medcpt_qwen36_l40.json"
MAX_TERMS = 24


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="how many patients to show")
    parser.add_argument("--out", default="llm_expansion_samples.json")
    args = parser.parse_args()

    from trialmatchai.config.config_loader import load_config
    from trialmatchai.interop.models import PatientProfile
    from trialmatchai.matching.query_expansion import build_first_level_expander
    from trialmatchai.matching.retrieval.first_level_planner import parse_llm_query_expansion

    cfg = load_config(BASE_CFG)
    first_level = dict(cfg["search"].get("first_level", {}))
    first_level.update({"llm_expansion_enabled": True, "llm_max_terms": MAX_TERMS})
    cfg["search"]["first_level"] = first_level

    expander = build_first_level_expander(cfg)
    if expander is None:
        raise SystemExit("expander could not be built")

    samples = []
    for summary_path in sorted(SUMMARIES.glob("*.json"))[: args.n]:
        pid = summary_path.stem
        profile_path = PROFILES / f"{pid}.json"
        if not profile_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        profile = PatientProfile.model_validate_json(profile_path.read_text())

        raw = expander.expand_first_level_queries(profile=profile, matching_summary=summary)
        parsed = parse_llm_query_expansion(raw, max_terms=MAX_TERMS)

        print("=" * 78)
        print(f"PATIENT {pid}")
        print("=" * 78)
        print("  what the expander reads:")
        for key in ("main_conditions", "other_conditions"):
            vals = [str(v) for v in (summary.get(key) or [])][:6]
            if vals:
                print(f"    {key}: {'; '.join(vals)}")
        narrative = " ".join(str(s) for s in (summary.get("patient_narrative") or []))[:300]
        if narrative:
            print(f"    narrative: {narrative}...")

        print("\n  raw expansion (six fields):")
        for field, values in (raw.items() if isinstance(raw, dict) else []):
            print(f"    {field:24s} {values}")

        flat = [
            *parsed.primary_queries,
            *parsed.disease_aliases,
            *parsed.broader_queries,
            *parsed.biomarker_queries,
            *parsed.treatment_queries,
        ]
        print(f"\n  -> channel terms actually searched (cap {MAX_TERMS}, weight 0.5): {len(flat)}")
        for term in flat:
            print(f"       {term}")
        if parsed.discarded_or_uncertain:
            print(f"  -> deliberately NOT searched: {parsed.discarded_or_uncertain}")
        print()
        samples.append({"patient": pid, "raw": raw, "searched": flat,
                        "discarded": parsed.discarded_or_uncertain})

    Path(args.out).write_text(json.dumps(samples, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
