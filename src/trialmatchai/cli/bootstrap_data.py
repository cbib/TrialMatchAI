from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import requests

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
) -> None:
    data_dir = root / "data"
    models_dir = root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)

    criteria_dir = data_dir / "processed_criteria"
    if force or not _extract_complete(criteria_dir):
        criteria_dir.mkdir(parents=True, exist_ok=True)
        for index in range(criteria_chunks):
            chunk_name = f"{CHUNK_PREFIX}_{index}.zip"
            chunk_path = data_dir / chunk_name
            _download_if_missing(
                f"{criteria_base_url.rstrip('/')}/{chunk_name}?download=1",
                chunk_path,
            )
            _verify_sha256(
                chunk_path,
                os.getenv(f"TRIALMATCHAI_CRITERIA_PART_{index}_SHA256"),
            )
            _safe_extract_zip(chunk_path, criteria_dir)
        _mark_extract_complete(criteria_dir)

    processed_trials_dir = data_dir / "processed_trials"
    if force or not _extract_complete(processed_trials_dir):
        processed_archive = data_dir / PROCESSED_TRIALS_ARCHIVE
        _download_if_missing(data_url, processed_archive)
        _verify_sha256(
            processed_archive, os.getenv("TRIALMATCHAI_PROCESSED_TRIALS_SHA256")
        )
        _safe_extract_tar_gz(processed_archive, data_dir)
        _mark_extract_complete(processed_trials_dir)

    if with_models:
        models_dir.mkdir(parents=True, exist_ok=True)
        if force or not _extract_complete(models_dir):
            models_archive = data_dir / MODELS_ARCHIVE
            _download_if_missing(models_url, models_archive)
            _verify_sha256(models_archive, os.getenv("TRIALMATCHAI_MODELS_SHA256"))
            _safe_extract_tar_gz(models_archive, models_dir)
            _mark_extract_complete(models_dir)

    if finetune_data:
        finetune_dir = data_dir / "finetune"
        if force or not _extract_complete(finetune_dir):
            finetune_dir.mkdir(parents=True, exist_ok=True)
            finetune_archive = data_dir / FINETUNE_ARCHIVE
            _download_if_missing(finetune_data_url, finetune_archive)
            _verify_sha256(
                finetune_archive, os.getenv("TRIALMATCHAI_FINETUNE_DATA_SHA256")
            )
            _safe_extract_zip(finetune_archive, finetune_dir)
            _mark_extract_complete(finetune_dir)

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

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


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


def _has_entries(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


_EXTRACT_MARKER = ".bootstrap_complete"


def _extract_complete(path: Path) -> bool:
    """True only when a prior extract wrote its completion sentinel.

    Presence of *some* entries is not proof: a killed extract leaves a partial tree
    that would otherwise bake a truncated corpus into every later run.
    """
    return (path / _EXTRACT_MARKER).exists()


def _mark_extract_complete(path: Path) -> None:
    (path / _EXTRACT_MARKER).write_text("ok\n", encoding="utf-8")


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
