#!/usr/bin/env python
"""Benchmark a first-level trial-retrieval embedder against the TREC Clinical Trials qrels.

Builds a trials-only LanceDB index with the chosen embedder (re-embedding the qrels corpus),
then runs first-level retrieval per patient and reports recall@k -- for both grade-2 (eligible)
and grade-1+2 (relevant, TrialGPT's definition). To isolate the *retrieval* embedder as the only
variable, concept-linking (the concept_synonym channel) is held fixed on a reference embedder
(bge-m3) regardless of which embedder is under test.

Non-destructive: each embedder writes benchmarks/embedders/<name>.json and builds its index at a
separate path, so bge-m3's numbers and index are never overwritten. The same harness will extend
to reranker/CoT models later.

Usage:
    python scripts/benchmark_embedder.py \
        --embedder medcpt \
        --registry benchmarks/embedders/registry.json \
        --base-config <config.json> \
        --data-dir <data_dir with patients/, trec/> \
        --processed-trials data/processed_trials \
        --index-root /tmp/bench_index \
        --tracks 21 22 \
        --out benchmarks/embedders/medcpt.json
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.entities import build_entity_annotator
from trialmatchai.main import _load_patient_inputs, run_first_level_search
from trialmatchai.models.embedding import build_embedder
from trialmatchai.registry.preparation import prepare_trial_document
from trialmatchai.schemas.phenopacket import Keywords
from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.trec import qrels as q
from trialmatchai.trec.corpus import resolve_tracks
from trialmatchai.trec.runner import _track_config

KS = [100, 200, 300, 500, 700, 1000, 2000]
# Held-fixed reference embedder for concept-linking (so only the retrieval embedder varies).
REFERENCE_EMBEDDER = {
    "backend": "hf",
    "model_name": "BAAI/bge-m3",
    "pooling": "mean",
    "max_length": 512,
    "normalize": True,
}
# Benchmark retrieval depth/width (current committed defaults) -- identical for every embedder.
PER_CHANNEL_SIZE = 600
MAX_TRIALS = 2000


def build_trials_index(embedder, processed_dir: Path, nct_ids: set[str], index_path: Path, reuse: bool = False, metric: str = "cosine") -> int:
    """Re-embed the qrels-corpus trials with `embedder` and write a trials-only LanceDB index.

    ``metric`` sets the ANN index similarity ("cosine" or "dot"). With ``reuse=True`` an
    already-built index at ``index_path`` is kept as-is (returns -1), so a crashed/interrupted
    run can resume retrieval without the expensive re-embed."""
    if reuse and (index_path / "trials.lance").exists():
        return -1
    backend = LanceDBSearchBackend(db_path=str(index_path), vector_metric=metric)
    docs = []
    missing = 0
    for nct in sorted(nct_ids):
        f = processed_dir / f"{nct}.json"
        if not f.exists():
            missing += 1
            continue
        doc = json.loads(f.read_text())
        doc.setdefault("nct_id", nct)
        docs.append(prepare_trial_document(doc, embedder))
    if not docs:
        raise SystemExit(f"No processed trials found under {processed_dir} for this track's qrels.")
    if missing:
        # These corpus trials can never be retrieved, yet recall denominators count judged-relevant
        # NCTs -- so recall is capped below 1.0. Surface the count rather than hide the deflation.
        print(f"  [warn] {missing}/{len(nct_ids)} qrels-corpus trials had no processed file "
              f"(unindexed; recall denominators still include any relevant ones)", flush=True)
    backend.index_trials(docs, recreate=True, metric=metric)
    return len(docs)


def recall_at(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    # Reuse the shared TREC recall helper so the benchmark can't drift from the eval definition.
    out = {}
    for k in KS:
        value = q.recall_at_k(ranked, relevant, k)
        if value is not None:
            out[f"recall@{k}"] = round(value, 4)
    return out


def mean_curves(curves: list[dict[str, float]]) -> dict[str, float]:
    agg: dict[str, list[float]] = {}
    for c in curves:
        for key, val in c.items():
            agg.setdefault(key, []).append(val)
    return {key: round(sum(v) / len(v), 4) for key, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True, help="Registry key of the embedder under test.")
    ap.add_argument("--registry", default="benchmarks/embedders/registry.json")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--processed-trials", default="data/processed_trials")
    ap.add_argument("--index-root", required=True, help="Where to build per-track indexes (node-local).")
    ap.add_argument("--tracks", nargs="+", default=["21", "22"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-index", action="store_true", help="Reuse an existing per-track index instead of rebuilding.")
    ap.add_argument("--vector-weight", type=float, default=0.5, help="Hybrid blend: score = (1-w)*text + w*vector (default 0.5).")
    ap.add_argument("--cutoffs", default=None, help="Comma-separated recall@k cutoffs (default 100,200,300,500,700,1000,2000).")
    args = ap.parse_args()

    if args.cutoffs:
        global KS
        KS = [int(x) for x in args.cutoffs.split(",")]

    registry = json.loads(Path(args.registry).read_text())
    if args.embedder not in registry:
        raise SystemExit(f"Unknown embedder '{args.embedder}'. Known: {[k for k in registry if not k.startswith('_')]}")
    emb_cfg = {k: v for k, v in registry[args.embedder].items() if not k.startswith("_")}
    metric = emb_cfg.get("similarity", "cosine")  # "cosine" or "dot" (read by us, ignored by build_embedder)

    base = load_config(args.base_config)
    processed_dir = Path(args.processed_trials)
    index_root = Path(args.index_root)

    print(f"=== embedder under test: {args.embedder} -> {emb_cfg.get('model_name')} "
          f"(query={emb_cfg.get('query_model_name', 'same')}) ===", flush=True)

    # Retrieval embedder (the variable) and the fixed concept-linking annotator (bge-m3).
    retrieval_embedder = build_embedder({"embedder": emb_cfg})
    ref_cfg = copy.deepcopy(base)
    ref_cfg["embedder"] = REFERENCE_EMBEDDER
    ref_embedder = build_embedder(ref_cfg)
    annotator = build_entity_annotator(ref_cfg, embedder=ref_embedder)
    print(f"embedder device={getattr(retrieval_embedder, 'device', '?')} "
          f"concept-linking held on {REFERENCE_EMBEDDER['model_name']}", flush=True)

    results: dict[str, object] = {
        "embedder": args.embedder,
        "model_name": emb_cfg.get("model_name"),
        "query_model_name": emb_cfg.get("query_model_name"),
        "per_channel_size": PER_CHANNEL_SIZE,
        "max_trials": MAX_TRIALS,
        "similarity": metric,
        "vector_weight": args.vector_weight,
        "tracks": {},
    }

    for spec in resolve_tracks(args.tracks, data_dir=Path(args.data_dir), results_root=Path(".")):
        tcfg = _track_config(base, spec)
        qr = q.parse_qrels(q.download_qrels(spec.key, spec.trec_dir / "qrels"), spec.id_prefix)
        corpus = q.corpus_ncts(qr)

        idx_path = index_root / f"search_{spec.key}_{args.embedder}"
        t0 = time.time()
        n = build_trials_index(retrieval_embedder, processed_dir, corpus, idx_path, reuse=args.reuse_index, metric=metric)
        if n < 0:
            print(f"[{spec.key}] reusing existing index -> {idx_path}", flush=True)
        else:
            print(f"[{spec.key}] indexed {n} trials with {args.embedder} ({metric}) in {time.time()-t0:.0f}s -> {idx_path}", flush=True)
        backend = LanceDBSearchBackend(db_path=str(idx_path), vector_metric=metric, vector_weight=args.vector_weight)

        # Retrieval config: fixed depth/width, hard filters on, no (unwired) llm channel.
        cfg = copy.deepcopy(tcfg)
        s = cfg["search"]
        s["max_trials_first_level"] = MAX_TRIALS
        fl = s["first_level"]
        fl["max_trials"] = MAX_TRIALS
        fl["per_channel_size"] = PER_CHANNEL_SIZE
        fl["hard_filters"] = ["age", "sex", "overall_status"]
        fl["llm_expansion_enabled"] = False

        patients = _load_patient_inputs(cfg)
        g2_curves, g12_curves = [], []
        timed = 0
        sidecar = tempfile.mkdtemp(prefix=f"flbench_{spec.key}_")
        for profile, summary in patients:
            qid = profile.patient_id
            if qid not in qr:
                continue
            g2 = {t for t, g in qr[qid].items() if g == 2}
            g12 = {t for t, g in qr[qid].items() if g in (1, 2)}
            if not g2 and not g12:
                continue
            kw = Keywords.model_validate(summary).model_dump()
            pinfo = dict(summary)
            pinfo["patient_narrative"] = summary.get("patient_narrative", [])
            t = time.time()
            res = run_first_level_search(kw, sidecar, pinfo, annotator, retrieval_embedder, cfg, backend, patient_profile=profile)
            if timed < 2:
                timed += 1
                print(f"  [timing] {spec.key} {qid}: first-level {time.time()-t:.1f}s", flush=True)
            ranked = res[0] if res else []
            if g2:
                g2_curves.append(recall_at(ranked, g2))
            if g12:
                g12_curves.append(recall_at(ranked, g12))
            # Running read on long runs: print (and checkpoint) the mean recall so far so a
            # partial result is visible without waiting for all patients.
            n_done = len(g2_curves)
            if n_done <= 3 or n_done % 5 == 0:
                rc = mean_curves(g2_curves)
                print(f"  [running] {spec.key} {n_done} pts: grade2 "
                      f"recall@500={rc.get('recall@500')} @1000={rc.get('recall@1000')}", flush=True)
                # Checkpoint the WHOLE result so far (already-finished tracks + this running one),
                # so a mid-run crash doesn't lose completed tracks.
                partial = dict(results)
                partial["partial"] = True
                partial["tracks"] = {
                    **results["tracks"],
                    spec.key: {"n_patients_grade2": n_done,
                               "n_patients_grade1and2": len(g12_curves),
                               "recall_grade2": rc,
                               "recall_grade1and2": mean_curves(g12_curves)},
                }
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(partial, indent=2))

        # grade-2 and grade-1+2 recall are averaged over their own patient populations (a topic can
        # have grade-1 judgments but no grade-2), so report each count rather than one shared total.
        track = {
            "n_patients_grade2": len(g2_curves),
            "n_patients_grade1and2": len(g12_curves),
            "recall_grade2": mean_curves(g2_curves),
            "recall_grade1and2": mean_curves(g12_curves),
        }
        results["tracks"][spec.key] = track
        print(f"[{spec.key}] grade2   {track['recall_grade2']}", flush=True)
        print(f"[{spec.key}] grade1+2 {track['recall_grade1and2']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"=== BENCHMARK DONE -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
