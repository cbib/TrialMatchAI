"""`trialmatchai report` — render a self-contained HTML match report.

Regenerates a portable ``report.html`` from a patient's existing results
(``ranked_trials.json`` + per-trial evaluations), without re-running matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trialmatchai.config.config_loader import normalize_config_paths, resolve_config_path
from trialmatchai.interop.exporters.html_report import profile_to_html_report
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

    written = 0
    for pdir in patient_dirs:
        if not (pdir / "ranked_trials.json").exists():
            logger.warning("Skipping %s: no ranked_trials.json", pdir.name)
            continue
        html = profile_to_html_report(pdir, summary_dir=summary_dir, trial_meta_folders=meta_folders)
        out_path = Path(args.out) if args.out else pdir / "report.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Wrote %s", out_path)
        written += 1

    if not written:
        logger.error("No reports generated (no patient result dirs with ranked_trials.json under %s).", output_dir)
        return 1
    logger.info("Generated %s report(s).", written)
    return 0
