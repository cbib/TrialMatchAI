"""A repeatable synthetic CLI walkthrough using the real CPU search pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from importlib import resources
from pathlib import Path

from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.integrity import verify_manifest, write_manifest

_MARKER = ".trialmatchai-demo.json"
_FIXTURE_VERSION = 1


def _create_workspace(root: Path) -> dict:
    # Build beside the destination so publishing is one same-filesystem rename.
    # A failed or interrupted setup leaves the requested directory empty; retry
    # can start normally without trusting partially written ownership markers.
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-init-", dir=root.parent) as temp:
        staging = Path(temp)
        config = _write_workspace(root, destination=staging)
        staging.replace(root)
    return config


def _write_workspace(root: Path, *, destination: Path) -> dict:
    from trialmatchai.config.settings import TrialMatchSettings

    package = resources.files("trialmatchai")
    config = json.loads(package.joinpath("config/config.json").read_text(encoding="utf-8"))
    data = root / "data"
    config["paths"] = {"output_dir": str(root / "results"), "trials_json_folder": str(data / "trials_jsons")}
    config["patient_inputs"].update({
        "raw_dir": str(data / "patients/raw"), "profile_dir": str(data / "patients/profiles"),
        "summary_dir": str(data / "patients/summaries"), "strict_validation": True,
    })
    config["entity_extraction"].update({
        "backend": "disabled", "model_name": "demo-unused", "device": "cpu",
        "schema_path": str(package.joinpath("entity_schemas/trialmatchai.yaml")),
    })
    config["concept_linker"].update({"enabled": False, "db_path": str(data / "concepts")})
    config["search_backend"].update({"db_path": str(data / "search"), "candidate_limit": 50})
    config["embedder"] = {"backend": "hashing", "model_name": "hashing", "use_gpu": False,
                          "hashing_dimensions": 64, "normalize": True}
    config["model"].update({"base_model": "demo-unused", "reranker_model_path": "demo-unused",
                           "cot_adapter_path": None, "reranker_adapter_path": ""})
    config["global"]["device"] = "cpu"
    config["LLM_reranker"]["enabled"] = False
    config["rag"]["enabled"] = False
    config["use_cot_reasoning"] = False
    config["query_expansion"]["enabled"] = False
    config["search"].update({"mode": "bm25", "second_level_keep_divisor": 1})
    config["registry"].update({"raw_dir": str(data / "registry/raw"),
                              "reports_dir": str(data / "registry/reports"),
                              "manifest_path": str(data / "registry/manifest.jsonl")})
    # The demo deliberately ignores deployment env overrides and .env files.
    # Its files and database must stay inside this synthetic workspace.
    config = TrialMatchSettings.model_validate(config).to_dict()
    write_json_file(config, str(destination / "config.json"))
    for number, condition, title, minimum, maximum in (
        (1, "lung cancer", "SYNTHETIC adult lung cancer study", "18 Years", "120 Years"),
        (2, "lung cancer", "SYNTHETIC pediatric lung cancer study", "0 Years", "12 Years"),
        (3, "diabetes", "SYNTHETIC adult diabetes study", "18 Years", "120 Years"),
    ):
        nct_id = f"NCT{number:08d}"
        write_json_file({
            "nct_id": nct_id, "brief_title": title, "brief_summary": f"Synthetic research on {condition}.",
            "condition": condition, "overall_status": "RECRUITING", "gender": "All",
            "minimum_age": minimum, "maximum_age": maximum,
            "eligibility_criteria": f"Inclusion Criteria:\n- Diagnosis of {condition}.\n- Age within the study limits.",
        }, str(destination / "data/trials_jsons" / f"{nct_id}.json"))
    write_json_file({"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {"resourceType": "Patient", "id": "demo-patient", "gender": "female", "birthDate": "1980-01-01"}},
        {"resource": {"resourceType": "Condition", "id": "demo-condition",
                      "subject": {"reference": "Patient/demo-patient"}, "code": {"text": "lung cancer"}}},
    ]}, str(destination / "patient.fhir.json"))
    write_json_file({"fixture_version": _FIXTURE_VERSION}, str(destination / _MARKER))
    # Covers the initial fixture/config inputs, not the runtime outputs added later.
    write_manifest(destination)
    return config


def run_demo(workdir: str | Path | None = None, *, resume: bool = False) -> dict:
    root = Path(workdir).resolve() if workdir is not None else Path(tempfile.mkdtemp(prefix="trialmatchai-demo-"))
    if resume:
        marker = root / _MARKER
        if not marker.is_file() or json.loads(marker.read_text()) != {"fixture_version": _FIXTURE_VERSION}:
            raise ValueError("--resume requires an existing TrialMatchAI demo workspace of this version")
        verify_manifest(root / "SHA256SUMS")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        # A copied workspace's absolute config paths must not write back to its origin.
        if config.get("paths", {}).get("output_dir") != str(root / "results"):
            raise ValueError("Demo workspace moved; create a new demo in an empty directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Directory is not empty: {root}. Choose an empty directory or use --resume for an existing demo.")
        root.mkdir(parents=True, exist_ok=True)
        config = _create_workspace(root)

    from trialmatchai.pipeline import StageContext, run_pipeline

    print(f"Synthetic CPU demo: {root}\nThis run tests retrieval and reporting; no eligibility model is used.", flush=True)
    ctx = StageContext(
        config=config, processed_trials_folder=root / "data/processed_trials",
        processed_criteria_folder=root / "data/processed_criteria",
        inputs=[str(root / "patient.fhir.json")], input_format="fhir", with_entities=False,
    )
    rc = run_pipeline(ctx, only=["prepare", "index", "ingest", "match"])
    if rc:
        raise RuntimeError(f"Demo pipeline failed with exit code {rc}")
    patient_dir = root / "results/demo-patient"
    ranked = json.loads((patient_dir / "ranked_trials.json").read_text())["RankedTrials"]
    result = {
        "schema_version": 1, "mode": "synthetic_retrieval_only", "workspace": str(root),
        "config": str(root / "config.json"), "report": str(patient_dir / "report.html"),
        "ranked_trial_ids": [row["TrialID"] for row in ranked], "resumed": resume,
    }
    if not ranked or not Path(result["report"]).is_file():
        raise RuntimeError("Demo did not produce a non-empty ranking and HTML report")
    write_json_file(result, str(root / "demo-result.json"))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trialmatchai demo", description=__doc__)
    parser.add_argument("--workdir", type=Path, help="Empty workspace directory (default: a new temporary directory)")
    parser.add_argument("--resume", action="store_true", help="Resume an existing, unmodified demo workspace")
    args = parser.parse_args(argv)
    if args.resume and args.workdir is None:
        parser.error("--resume requires --workdir")
    workdir = args.workdir
    try:
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="trialmatchai-demo-"))
        result = run_demo(workdir, resume=args.resume)
    except KeyboardInterrupt:
        print("Demo interrupted.", file=sys.stderr)
        if workdir is not None:
            resume_flag = " --resume" if (workdir / "SHA256SUMS").is_file() else ""
            print(f"To retry: trialmatchai demo --workdir {shlex.quote(str(workdir))}{resume_flag}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {result['report']}\nRun details: {Path(result['workspace']) / 'demo-result.json'}")
    print(f"To resume: trialmatchai demo --workdir {shlex.quote(result['workspace'])} --resume")
    return 0
