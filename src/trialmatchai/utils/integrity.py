"""SHA-256 manifests for transferred artifacts (not publisher authentication)."""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path, PurePosixPath

from trialmatchai.utils.file_utils import write_text_file


def normalize_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected a SHA-256 digest containing exactly 64 hexadecimal characters")
    return value.lower()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    expected = normalize_sha256(expected)
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(f"Checksum mismatch for {Path(path).name}: expected {expected}, got {actual}")
    return actual


def _validate_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name or path.is_absolute() or "\\" in name or ":" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Unsafe artifact path in checksum manifest: {name!r}")
    return name


def read_manifest(path: str | Path) -> dict[str, str]:
    """Read the portable GNU sha256sum text format; reject ambiguity/traversal."""
    entries: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError(f"Invalid SHA-256 manifest line {number}")
        name = _validate_name(match.group(2))
        if name in entries:
            raise ValueError(f"Duplicate artifact in checksum manifest: {name}")
        entries[name] = normalize_sha256(match.group(1))
    if not entries:
        raise ValueError("Checksum manifest contains no artifacts")
    return entries


def _artifact_path(root: Path, name: str) -> Path:
    candidate = root
    for part in PurePosixPath(_validate_name(name)).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Artifact must not be a symlink: {name}")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Artifact escapes its directory: {name}")
    if not candidate.is_file():
        raise ValueError(f"Artifact is missing or is not a regular file: {name}")
    return candidate


def write_manifest(directory: str | Path, *, filename: str = "SHA256SUMS") -> Path:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Artifact directory does not exist: {root}")
    if len(PurePosixPath(_validate_name(filename)).parts) != 1:
        raise ValueError("Manifest filename must be a single filename")
    manifest = root / filename
    if manifest.is_symlink():
        raise ValueError("Checksum manifest must not be a symlink")
    lines = []
    for path in sorted(root.rglob("*")):
        if path == manifest:
            continue
        if path.is_symlink():
            raise ValueError(f"Artifact must not be a symlink: {path.relative_to(root)}")
        if path.is_file():
            name = _validate_name(path.relative_to(root).as_posix())
            lines.append(f"{sha256_file(path)}  {name}")
    if not lines:
        raise ValueError("No artifacts to checksum")
    write_text_file(lines, str(manifest))
    return manifest


def verify_manifest(
    manifest: str | Path, *, directory: str | Path | None = None, require_exact: bool = False,
) -> list[str]:
    manifest = Path(manifest).resolve()
    root = Path(directory).resolve() if directory is not None else manifest.parent
    entries = read_manifest(manifest)
    for name, expected in entries.items():
        verify_sha256(_artifact_path(root, name), expected)
    if require_exact:
        found = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if (path.is_file() or path.is_symlink()) and path != manifest
        }
        if found != set(entries):
            raise ValueError(f"Unlisted artifacts: {sorted(found - set(entries))}")
    return sorted(entries)
