from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import requests

from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.integrity import normalize_sha256, read_manifest, verify_sha256

DATA_URL = "https://zenodo.org/records/15516900/files/processed_trials.tar.gz?download=1"
MODELS_URL = "https://zenodo.org/records/15516900/files/models.tar.gz?download=1"
CRITERIA_ZIP_BASE_URL = "https://zenodo.org/records/15516900/files"
# Fine-tuning datasets (CoT/reranker/NER JSONL) live on the paper's deposit.
FINETUNE_DATA_URL = (
    "https://zenodo.org/records/15045515/files/finetuning_datasets.zip?download=1"
)
CHUNK_PREFIX = "criteria_part"
CHUNK_COUNT = 6
PROCESSED_TRIALS_ARCHIVE = "processed_trials.tar.gz"
MODELS_ARCHIVE = "models.tar.gz"
FINETUNE_ARCHIVE = "finetuning_datasets.zip"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare TrialMatchAI data and model artifacts"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Runtime root for data/ and models/; defaults to repository root or current directory",
    )
    parser.add_argument(
        "--data-url",
        default=DATA_URL,
        help="processed_trials.tar.gz URL",
    )
    parser.add_argument(
        "--models-url",
        default=MODELS_URL,
        help="models.tar.gz URL",
    )
    parser.add_argument(
        "--criteria-base-url",
        default=CRITERIA_ZIP_BASE_URL,
        help="Base URL containing criteria_part_<n>.zip chunks",
    )
    parser.add_argument(
        "--criteria-chunks",
        type=int,
        default=CHUNK_COUNT,
        help="Number of criteria zip chunks to download",
    )
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="Also fetch the fine-tuned adapters from Zenodo into models/. Not needed by "
        "default: the adapters download from Hugging Face on first use.",
    )
    parser.add_argument(
        "--finetune-data",
        action="store_true",
        help="Also download the fine-tuning datasets (CoT/reranker/NER JSONL) to data/finetune/.",
    )
    parser.add_argument(
        "--finetune-data-url",
        default=FINETUNE_DATA_URL,
        help="finetuning_datasets.zip URL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract archives even when target directories already exist",
    )
    parser.add_argument(
        "--checksum-manifest", type=Path,
        help="Trusted SHA256SUMS file for requested archives; implies --require-checksums.",
    )
    parser.add_argument(
        "--require-checksums", action="store_true",
        help="Require SHA-256 for every requested archive before any download/extraction. "
        "Use --checksum-manifest or TRIALMATCHAI_*_SHA256 environment variables.",
    )
    args = parser.parse_args(argv)

    root = (args.root or _runtime_root()).resolve()
    bootstrap_data(
        root=root,
        data_url=args.data_url,
        models_url=args.models_url,
        criteria_base_url=args.criteria_base_url,
        criteria_chunks=args.criteria_chunks,
        with_models=args.with_models,
        finetune_data=args.finetune_data,
        finetune_data_url=args.finetune_data_url,
        force=args.force,
        checksum_manifest=args.checksum_manifest,
        require_checksums=args.require_checksums,
    )
    return 0


def bootstrap_data(
    *,
    root: Path,
    data_url: str = DATA_URL,
    models_url: str = MODELS_URL,
    criteria_base_url: str = CRITERIA_ZIP_BASE_URL,
    criteria_chunks: int = CHUNK_COUNT,
    with_models: bool = False,
    finetune_data: bool = False,
    finetune_data_url: str = FINETUNE_DATA_URL,
    force: bool = False,
    checksum_manifest: str | Path | None = None,
    require_checksums: bool = False,
) -> None:
    if criteria_chunks < 1:
        raise ValueError("criteria_chunks must be at least 1")
    require_checksums = require_checksums or checksum_manifest is not None
    supplied = read_manifest(checksum_manifest) if checksum_manifest is not None else {}
    env_names = {
        PROCESSED_TRIALS_ARCHIVE: "TRIALMATCHAI_PROCESSED_TRIALS_SHA256",
        **{f"{CHUNK_PREFIX}_{i}.zip": f"TRIALMATCHAI_CRITERIA_PART_{i}_SHA256" for i in range(criteria_chunks)},
    }
    if with_models:
        env_names[MODELS_ARCHIVE] = "TRIALMATCHAI_MODELS_SHA256"
    if finetune_data:
        env_names[FINETUNE_ARCHIVE] = "TRIALMATCHAI_FINETUNE_DATA_SHA256"
    checksums: dict[str, str | None] = {}
    for name, env in env_names.items():
        value = supplied.get(name) or os.getenv(env)
        checksums[name] = normalize_sha256(value) if value else None
    missing = [name for name, value in checksums.items() if value is None]
    if require_checksums and missing:
        raise ValueError(f"Missing required SHA-256 checksums: {', '.join(missing)}")

    def stage_checksums(names):
        return {name: checksums[name] for name in names if checksums[name] is not None}

    def complete(path, expected):
        return _extract_complete(path, expected_checksums=expected if require_checksums else None)

    data_dir = root / "data"
    models_dir = root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)

    criteria_dir = data_dir / "processed_criteria"
    criteria_sums = stage_checksums(f"{CHUNK_PREFIX}_{i}.zip" for i in range(criteria_chunks))
    if force or not complete(criteria_dir, criteria_sums):
        criteria_dir.mkdir(parents=True, exist_ok=True)
        (criteria_dir / _EXTRACT_MARKER).unlink(missing_ok=True)
        for index in range(criteria_chunks):
            chunk_name = f"{CHUNK_PREFIX}_{index}.zip"
            chunk_path = data_dir / chunk_name
            _download_if_missing(
                f"{criteria_base_url.rstrip('/')}/{chunk_name}?download=1",
                chunk_path,
            )
            _verify_sha256(
                chunk_path,
                checksums[chunk_name],
            )
            _safe_extract_zip(chunk_path, criteria_dir)
        _mark_extract_complete(criteria_dir, checksums=criteria_sums)

    processed_trials_dir = data_dir / "processed_trials"
    trial_sums = stage_checksums([PROCESSED_TRIALS_ARCHIVE])
    if force or not complete(processed_trials_dir, trial_sums):
        (processed_trials_dir / _EXTRACT_MARKER).unlink(missing_ok=True)
        processed_archive = data_dir / PROCESSED_TRIALS_ARCHIVE
        _download_if_missing(data_url, processed_archive)
        _verify_sha256(
            processed_archive, checksums[PROCESSED_TRIALS_ARCHIVE]
        )
        _safe_extract_tar_gz(processed_archive, data_dir)
        _mark_extract_complete(processed_trials_dir, checksums=trial_sums)

    if with_models:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_sums = stage_checksums([MODELS_ARCHIVE])
        if force or not complete(models_dir, model_sums):
            (models_dir / _EXTRACT_MARKER).unlink(missing_ok=True)
            models_archive = data_dir / MODELS_ARCHIVE
            _download_if_missing(models_url, models_archive)
            _verify_sha256(models_archive, checksums[MODELS_ARCHIVE])
            _safe_extract_tar_gz(models_archive, models_dir)
            _mark_extract_complete(models_dir, checksums=model_sums)

    if finetune_data:
        finetune_dir = data_dir / "finetune"
        finetune_sums = stage_checksums([FINETUNE_ARCHIVE])
        if force or not complete(finetune_dir, finetune_sums):
            finetune_dir.mkdir(parents=True, exist_ok=True)
            (finetune_dir / _EXTRACT_MARKER).unlink(missing_ok=True)
            finetune_archive = data_dir / FINETUNE_ARCHIVE
            _download_if_missing(finetune_data_url, finetune_archive)
            _verify_sha256(
                finetune_archive, checksums[FINETUNE_ARCHIVE]
            )
            _safe_extract_zip(finetune_archive, finetune_dir)
            _mark_extract_complete(finetune_dir, checksums=finetune_sums)

    _cleanup_archives(data_dir, criteria_chunks)


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        _info(f"{destination.name} already exists; skipping download.")
        return

    _info(f"Downloading {destination.name}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stream to .part and rename on success so a killed download never leaves a truncated
    # archive at the final path (checksum verification is optional, so can't catch it).
    partial = destination.with_name(destination.name + ".part")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with partial.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    partial.replace(destination)


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        _warn(f"No SHA-256 checksum configured for {path.name}; skipping verification.")
        return

    verify_sha256(path, expected)


def _safe_extract_tar_gz(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            _validated_target_path(target, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Archive contains an unsafe member: {member.name}")
        tar.extractall(target)


def _safe_extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            _validated_target_path(target, member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive contains an unsafe member: {member.filename}")
        zip_file.extractall(target)


def _validated_target_path(target: Path, member_name: str) -> Path:
    if not member_name:
        raise ValueError("Archive contains an empty path")
    member_path = Path(member_name)
    if member_path.is_absolute():
        raise ValueError(f"Archive contains an absolute path: {member_name}")

    resolved_target = target.resolve()
    resolved_member = (resolved_target / member_path).resolve()
    try:
        resolved_member.relative_to(resolved_target)
    except ValueError as exc:
        raise ValueError(f"Archive contains an unsafe path: {member_name}") from exc
    return resolved_member


def _cleanup_archives(data_dir: Path, criteria_chunks: int) -> None:
    for path in [
        data_dir / PROCESSED_TRIALS_ARCHIVE,
        data_dir / MODELS_ARCHIVE,
        data_dir / FINETUNE_ARCHIVE,
    ]:
        path.unlink(missing_ok=True)
    for index in range(criteria_chunks):
        (data_dir / f"{CHUNK_PREFIX}_{index}.zip").unlink(missing_ok=True)


_EXTRACT_MARKER = ".bootstrap_complete"


def _extract_complete(path: Path, *, expected_checksums: dict[str, str] | None = None) -> bool:
    """True only when a prior extract wrote its completion sentinel.

    Presence of *some* entries is not proof: a killed extract leaves a partial tree
    that would otherwise bake a truncated corpus into every later run.
    """
    marker = path / _EXTRACT_MARKER
    if expected_checksums is None:
        return marker.is_file()
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("schema") == 1 and data.get("sha256") == expected_checksums
    except (OSError, ValueError):
        return False


def _mark_extract_complete(path: Path, *, checksums: dict[str, str] | None = None) -> None:
    write_json_file({"schema": 1, "sha256": checksums or {}}, str(path / _EXTRACT_MARKER))


def _runtime_root() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _info(message: str) -> None:
    print(f"[INFO] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
