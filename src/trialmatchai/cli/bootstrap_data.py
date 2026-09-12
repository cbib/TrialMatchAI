from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import uuid
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
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

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _bootstrap_lock(root):
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        stages = [
            (data_dir / "processed_criteria", [
                (f"{CHUNK_PREFIX}_{i}.zip", f"{criteria_base_url.rstrip('/')}/{CHUNK_PREFIX}_{i}.zip?download=1", "zip")
                for i in range(criteria_chunks)
            ], False),
            (data_dir / "processed_trials", [(PROCESSED_TRIALS_ARCHIVE, data_url, "trials")], False),
        ]
        if with_models:
            stages.append((root / "models", [(MODELS_ARCHIVE, models_url, "tar")], True))
        if finetune_data:
            stages.append((data_dir / "finetune", [(FINETUNE_ARCHIVE, finetune_data_url, "zip")], False))
        for target, archives, shared in stages:
            _recover_publication(target)
            expected = {name: checksums[name] for name, _, _ in archives if checksums[name] is not None}
            if not force and _extract_complete(target, expected_checksums=expected if require_checksums else None):
                continue
            _replace_stage(target, archives, data_dir, checksums, expected, shared=shared)
        _cleanup_archives(data_dir, criteria_chunks)


@contextmanager
def _bootstrap_lock(root: Path):
    import fcntl

    fd = os.open(root / ".bootstrap.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Bootstrap lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another bootstrap is already running in {root}") from exc
        yield
    finally:
        os.close(fd)


def _download_verified(url: str, destination: Path, expected: str | None) -> None:
    # One replacement attempt; never weaken a trusted checksum to accept a file.
    for attempt in range(2):
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"Cached archive must be a regular file: {destination}")
        _download_if_missing(url, destination)
        try:
            _verify_sha256(destination, expected)
            return
        except ValueError:
            quarantine = destination.with_name(f".{destination.name}.corrupt-{uuid.uuid4().hex}")
            destination.replace(quarantine)
            _warn(f"Invalid archive retained at {quarantine}")
            if attempt:
                raise
            _info(f"Retrying {destination.name} with a fresh download.")


def _replace_stage(target, archives, data_dir, checksums, expected, *, shared):
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"Extraction destination must be a directory, not a link: {target}")
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-bootstrap-", dir=target.parent) as temp:
        staging = Path(temp) / "payload"
        staging.mkdir()
        for name, url, kind in archives:
            archive = data_dir / name
            _download_verified(url, archive, checksums[name])
            if kind == "zip":
                _safe_extract_zip(archive, staging)
            elif kind == "trials":
                wrapper = Path(temp) / "trials-archive"
                _safe_extract_tar_gz(archive, wrapper)
                children = list(wrapper.iterdir())
                if len(children) != 1 or children[0].name != "processed_trials" or not children[0].is_dir():
                    raise ValueError("Trial archive must contain only the processed_trials/ directory")
                children[0].replace(staging)
            else:
                _safe_extract_tar_gz(archive, staging)
        roots = sorted(path.name for path in staging.iterdir() if path.name != _EXTRACT_MARKER)
        if not roots:
            raise ValueError(f"Archive produced no artifacts for {target.name}")
        if shared and target.exists():
            _preserve_unmanaged_models(target, staging, roots)
        _mark_extract_complete(staging, checksums=expected, managed_roots=roots)
        _publish_stage(staging, target)


def _managed_roots(marker):
    if not isinstance(marker, dict) or marker.get("schema") != 2:
        return []
    roots = marker.get("managed_roots")
    if not isinstance(roots, list) or not roots:
        return []
    if any(not isinstance(name, str) or name in {"", ".", "..", _EXTRACT_MARKER}
           or "/" in name or "\\" in name for name in roots):
        return []
    return roots


def _preserve_unmanaged_models(target, staging, incoming):
    previous = _read_marker(target / _EXTRACT_MARKER)
    managed = set(_managed_roots(previous)) | set(incoming) | {_EXTRACT_MARKER}
    # Preserve independent top-level models. Replace whole owned adapter folders,
    # including removed files; old trees remain available in the recovery backup.
    for path in target.iterdir():
        if path.name in managed:
            continue
        destination = staging / path.name
        if path.is_symlink():
            destination.symlink_to(os.readlink(path))
        elif path.is_dir():
            shutil.copytree(path, destination, symlinks=True, copy_function=_link_or_copy)
        elif path.is_file():
            _link_or_copy(path, destination)
        else:
            raise ValueError(f"Cannot preserve non-regular model entry: {path}")


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def _previous_path(target):
    return target.with_name(f".{target.name}.bootstrap-previous")


def _recover_publication(target):
    previous = _previous_path(target)
    if not previous.exists() and not previous.is_symlink():
        return
    if previous.is_symlink() or not previous.is_dir():
        raise ValueError(f"Invalid bootstrap recovery directory: {previous}")
    if not target.exists() and not target.is_symlink():
        previous.replace(target)
        _info(f"Recovered the previous extraction at {target}")
    else:
        saved = Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent))
        previous.replace(saved)
        _info(f"Previous extraction retained at {saved}")


def _publish_stage(staging, target):
    previous = _previous_path(target)
    if target.exists():
        target.replace(previous)
    try:
        staging.replace(target)
    except BaseException:
        _recover_publication(target)
        raise
    _recover_publication(target)


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        _info(f"{destination.name} already exists; skipping download.")
        return

    _info(f"Downloading {destination.name}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stream to .part and rename on success so a killed download never leaves a truncated
    # archive at the final path (checksum verification is optional, so can't catch it).
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f"Partial download must be a regular file: {partial}")
    response = requests.get(url, stream=True, timeout=120)
    try:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    finally:
        response.close()
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


def _read_marker(marker: Path) -> dict | None:
    try:
        fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
                return None
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _extract_complete(path: Path, *, expected_checksums: dict[str, str] | None = None) -> bool:
    marker = path / _EXTRACT_MARKER
    try:
        if not stat.S_ISREG(marker.lstat().st_mode):
            return False
    except OSError:
        return False
    if expected_checksums is None:
        return True  # Legacy non-strict completion markers remain compatible.
    value = _read_marker(marker)
    roots = _managed_roots(value)
    return bool(roots) and value.get("sha256") == expected_checksums and all(
        (path / name).exists() and not (path / name).is_symlink() for name in roots
    )


def _mark_extract_complete(path: Path, *, checksums: dict[str, str] | None = None, managed_roots: list[str] | None = None) -> None:
    roots = managed_roots if managed_roots is not None else sorted(p.name for p in path.iterdir() if p.name != _EXTRACT_MARKER)
    write_json_file({"schema": 2, "sha256": checksums or {}, "managed_roots": roots}, str(path / _EXTRACT_MARKER))


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
