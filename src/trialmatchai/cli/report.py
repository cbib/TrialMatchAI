"""`trialmatchai report` — render a self-contained HTML match report.

Regenerates a portable ``report.html`` from a patient's existing results
(``ranked_trials.json`` + per-trial evaluations), without re-running matching.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from trialmatchai.config.config_loader import normalize_config_paths, resolve_config_path
from trialmatchai.interop.exporters.html_report import (
    profile_to_html_report,
    profile_to_model,
    render_unified_html,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="trialmatchai report",
        description="Render a self-contained HTML match report from a patient's results.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Patient id (its <output_dir>/<id> result directory).")
    group.add_argument("--all", action="store_true", help="Render a report for every patient result directory.")
    parser.add_argument("--config", default=None, help="Path to config.json.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (single --patient only; default: <patient_dir>/report.html).",
    )
    args = parser.parse_args()
    if args.out and args.all:
        parser.error("--out cannot be combined with --all")

    # The report renders existing results, so it only needs the path config —
    # not the full (model/embedder) config that load_config would validate.
    resolved = resolve_config_path(args.config)
    config = normalize_config_paths(json.loads(resolved.read_text(encoding="utf-8")), resolved)
    output_dir = Path(config["paths"]["output_dir"])
    summary_dir = config.get("patient_inputs", {}).get("summary_dir")
    trials_json = Path(config["paths"].get("trials_json_folder", "data/trials_jsons"))
    meta_folders = [trials_json.parent / "processed_trials", trials_json]

    if args.all:
        patient_dirs = sorted(p for p in output_dir.iterdir() if p.is_dir()) if output_dir.is_dir() else []
    else:
        patient_dirs = [output_dir / args.patient]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid_dirs = []
    for pdir in patient_dirs:
        if (pdir / "ranked_trials.json").exists():
            valid_dirs.append(pdir)
        else:
            logger.warning("Skipping %s: no ranked_trials.json", pdir.name)
    if not valid_dirs:
        logger.error("No patient result dirs with ranked_trials.json under %s.", output_dir)
        return 1

    if args.all:
        # One unified, self-contained report: a front page listing every patient
        # that drills into the per-patient view.
        models = []
        for pdir in valid_dirs:
            try:
                models.append(profile_to_model(
                    pdir, summary_dir=summary_dir, trial_meta_folders=meta_folders, generated_at=generated_at))
            except Exception:
                logger.exception("Skipping %s: failed to build report model", pdir.name)
        if not models:
            logger.error("No reports generated.")
            return 1
        out_path = output_dir / "index.html"
        out_path.write_text(render_unified_html(models, generated_at), encoding="utf-8")
        logger.info("Wrote unified report %s (%d patients).", out_path, len(models))
        return 0

    # single patient
    pdir = valid_dirs[0]
    html = profile_to_html_report(
        pdir, summary_dir=summary_dir, trial_meta_folders=meta_folders, generated_at=generated_at)
    out_path = Path(args.out) if args.out else pdir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0
