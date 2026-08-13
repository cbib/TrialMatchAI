#!/usr/bin/env python
"""Does the llm_expansion channel lift the first-level recall ceiling?

The depth sweep established the ceiling: recall plateaus at 0.9110 and the candidate pool
exhausts at ~2,525 trials, so ~9% of judged-relevant trials are never retrieved and no amount
of depth reaches them. New QUERY TERMS are one of the few things that can.

llm_expansion is a ninth retrieval channel (weight 0.5) whose terms come from an LLM reading
the patient -- disease aliases, broader categories, biomarker and treatment phrasings. Unlike
the other eight channels it is not limited to terms already in the record or derivable by
lookup.

Run one arm at a time; each arm PERSISTS its per-patient candidate lists, so an arm is never
re-run to answer a later question. When both arms' lists exist the comparison is computed
from disk at no cost -- including the number that decides whether iterating is worth it:
how many trials the new terms surface that the other channels never found, and how many of
those are relevant.

    python scripts/llm_expansion_ab.py --arm off
    python scripts/llm_expansion_ab.py --arm on
    python scripts/llm_expansion_ab.py --compare

Must be run as a FILE, not piped to `python -`: vLLM falls back to spawn multiprocessing once
CUDA is initialised, and spawn re-imports __main__, which fails for a stdin script.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

DEPTH = 4000  # where first-level recall plateaus; identical in both arms
PER_CHANNEL = 600
MAX_TERMS = 24  # 12 is tight: the budget is shared across five fields and spent in order
OUT_DIR = Path("llm_expansion_ab")
SUMMARIES = Path("data/patients/trec23/summaries")
PROFILES = Path("data/patients/trec23/profiles")
BASE_CFG = "src/trialmatchai/config/config_medcpt_qwen36_l40.json"
QRELS = Path("data/trec/qrels/qrels_23.txt")


def _base_config():
    from trialmatchai.config.config_loader import load_config

    cfg = load_config(BASE_CFG)
    cfg.setdefault("search_backend", {})["db_path"] = "data/search_medcpt_23"
    cfg.setdefault("embedder", {})["use_gpu"] = True
    cfg.setdefault("concept_linker", {})["db_path"] = "data/concepts_medcpt"
    return cfg


def _arm_config(base, enabled: bool):
    cfg = json.loads(json.dumps(base))
    first_level = dict(cfg["search"].get("first_level", {}))
    first_level.update(
        {
            "max_trials": DEPTH,
            "per_channel_size": PER_CHANNEL,
            "write_reports": False,
            "llm_expansion_enabled": enabled,
            "llm_max_terms": MAX_TERMS,
        }
    )
    cfg["search"]["first_level"] = first_level
    cfg["search"]["max_trials_first_level"] = DEPTH
    return cfg


def _patients(relevant):
    from trialmatchai.interop.models import PatientProfile

    out = []
    for summary_path in sorted(SUMMARIES.glob("*.json")):
        pid = summary_path.stem
        profile_path = PROFILES / f"{pid}.json"
        if pid not in relevant or not relevant[pid] or not profile_path.exists():
            continue
        try:
            out.append(
                (
                    pid,
                    json.loads(summary_path.read_text()),
                    PatientProfile.model_validate_json(profile_path.read_text()),
                )
            )
        except Exception as exc:  # a malformed profile must not abort the arm
            print(f"  skip {pid}: {exc}")
    return out


def run_arm(arm: str) -> None:
    from trialmatchai.entities import build_entity_annotator
    from trialmatchai.main import run_first_level_search
    from trialmatchai.matching.query_expansion import build_first_level_expander
    from trialmatchai.models.embedding.text_embedder import build_embedder
    from trialmatchai.search import build_search_backend
    from trialmatchai.trec.qrels import parse_qrels, relevant_ncts

    enabled = arm == "on"
    base = _base_config()
    cfg = _arm_config(base, enabled)

    backend = build_search_backend(base)
    embedder = build_embedder(base)
    annotator = build_entity_annotator(base, embedder=embedder)

    expander = None
    if enabled:
        expander = build_first_level_expander(cfg)
        if expander is None:
            raise SystemExit("llm_expansion enabled but the expander could not be built")

    relevant = relevant_ncts(parse_qrels(QRELS, "trec-2023"), threshold=1)
    patients = _patients(relevant)
    print(f"arm {arm}: {len(patients)} patients, depth={DEPTH}, max_terms={MAX_TERMS}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / f"{arm}_ids.json"
    # Persist incrementally: a crash in patient 30 must not discard the first 29.
    ids_by_patient = json.loads(dest.read_text()) if dest.exists() else {}

    for pid, summary, profile in patients:
        if pid in ids_by_patient:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = run_first_level_search(
                    summary,
                    tmp,
                    {"age": summary.get("age", "all"), "gender": summary.get("gender", "all")},
                    annotator,
                    embedder,
                    cfg,
                    backend,
                    patient_profile=profile,
                    llm_query_expander=expander,
                )
            except Exception as exc:
                print(f"    {pid} failed: {exc}", flush=True)
                continue
        if not result:
            continue
        ids_by_patient[pid] = result[0]
        dest.write_text(json.dumps(ids_by_patient))
        print(f"    {pid}: {len(result[0])}", flush=True)

    print(f"arm {arm} done -> {dest} ({len(ids_by_patient)} patients)")


def compare() -> None:
    from trialmatchai.trec.qrels import parse_qrels, recall_at_k, relevant_ncts

    qrels = parse_qrels(QRELS, "trec-2023")
    relevant = relevant_ncts(qrels, threshold=1)
    eligible = relevant_ncts(qrels, threshold=2)

    arms = {}
    for arm in ("off", "on"):
        path = OUT_DIR / f"{arm}_ids.json"
        if not path.exists():
            print(f"missing {path} — run: python scripts/llm_expansion_ab.py --arm {arm}")
            return
        arms[arm] = json.loads(path.read_text())

    common = sorted(set(arms["off"]) & set(arms["on"]))
    print(f"\npaired patients: {len(common)}\n")
    print(f"  {'arm':>5s} {'retrieved':>10s} {'recall(rel)':>12s} {'recall(elig)':>13s}")
    stats = {}
    for arm in ("off", "on"):
        rec = [recall_at_k(arms[arm][p], relevant[p], len(arms[arm][p])) for p in common]
        el = [
            recall_at_k(arms[arm][p], eligible[p], len(arms[arm][p]))
            for p in common
            if eligible.get(p)
        ]
        size = [len(arms[arm][p]) for p in common]
        stats[arm] = (sum(size) / len(size), sum(rec) / len(rec), sum(el) / len(el) if el else float("nan"))
        print(f"  {arm:>5s} {stats[arm][0]:10.0f} {stats[arm][1]:12.4f} {stats[arm][2]:13.4f}")

    print(
        f"\n  DELTA (on-off): retrieved {stats['on'][0] - stats['off'][0]:+.0f}   "
        f"recall(rel) {stats['on'][1] - stats['off'][1]:+.4f}   "
        f"recall(elig) {stats['on'][2] - stats['off'][2]:+.4f}"
    )

    # The number that decides whether ITERATING is worth building.
    new_total = new_rel = 0
    wins = 0
    for pid in common:
        fresh = set(arms["on"][pid]) - set(arms["off"][pid])
        hit = fresh & relevant[pid]
        new_total += len(fresh)
        new_rel += len(hit)
        wins += bool(hit)
    print(
        f"\n  surfaced ONLY by llm_expansion: {new_total} trials, {new_rel} judged-relevant "
        f"({100 * new_rel / max(1, new_total):.1f}% precision)"
    )
    print(f"  per patient: {new_total / len(common):.0f} new, {new_rel / len(common):.1f} relevant")
    print(f"  patients where it found >=1 new relevant trial: {wins}/{len(common)}")
    print("\n  That per-round yield is the stopping signal an iterative expander would use.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("off", "on"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.arm:
        run_arm(args.arm)
    if args.compare or not args.arm:
        compare()


if __name__ == "__main__":
    main()
