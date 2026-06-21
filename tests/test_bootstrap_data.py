from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from trialmatchai.cli.bootstrap_data import (
    MODELS_ARCHIVE,
    PROCESSED_TRIALS_ARCHIVE,
    bootstrap_data,
    _safe_extract_tar_gz,
    _safe_extract_zip,
    _verify_sha256,
)


def test_bootstrap_data_uses_existing_archives_and_removes_them(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    processed_archive = data_dir / PROCESSED_TRIALS_ARCHIVE
    models_archive = data_dir / MODELS_ARCHIVE
    criteria_archive = data_dir / "criteria_part_0.zip"

    _write_tar_gz(
        processed_archive,
        {"processed_trials/NCT000001.json": b'{"nct_id": "NCT000001"}'},
    )
    _write_tar_gz(models_archive, {"demo-model/config.json": b"{}"})
    _write_zip(criteria_archive, {"criterion.txt": "Age >= 18"})

    monkeypatch.setenv(
        "TRIALMATCHAI_PROCESSED_TRIALS_SHA256", _sha256(processed_archive)
    )
    monkeypatch.setenv("TRIALMATCHAI_MODELS_SHA256", _sha256(models_archive))
    monkeypatch.setenv("TRIALMATCHAI_CRITERIA_PART_0_SHA256", _sha256(criteria_archive))

    bootstrap_data(
        root=tmp_path,
        data_url="https://example.invalid/processed_trials.tar.gz",
        models_url="https://example.invalid/models.tar.gz",
        criteria_base_url="https://example.invalid",
        criteria_chunks=1,
    )

    assert (tmp_path / "data/processed_trials/NCT000001.json").exists()
    assert (tmp_path / "data/processed_criteria/criterion.txt").exists()
    assert (tmp_path / "models/demo-model/config.json").exists()
    assert not processed_archive.exists()
    assert not models_archive.exists()
    assert not criteria_archive.exists()

    bootstrap_data(
        root=tmp_path,
        data_url="https://example.invalid/missing-processed_trials.tar.gz",
        models_url="https://example.invalid/missing-models.tar.gz",
        criteria_base_url="https://example.invalid/missing",
        criteria_chunks=1,
    )


def test_verify_sha256_rejects_mismatches(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("contents")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_sha256(path, "0" * 64)


def test_tar_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"bad"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match="unsafe path"):
        _safe_extract_tar_gz(archive, tmp_path / "target")


def test_tar_extraction_rejects_links(tmp_path):
    archive = tmp_path / "unsafe-link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="unsafe member"):
        _safe_extract_tar_gz(archive, tmp_path / "target")


def test_zip_extraction_rejects_symlinks(tmp_path):
    archive = tmp_path / "unsafe-link.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        zip_file.writestr(info, "/etc/passwd")

    with pytest.raises(ValueError, match="unsafe member"):
        _safe_extract_zip(archive, tmp_path / "target")


def _write_tar_gz(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zip_file:
        for name, data in files.items():
            zip_file.writestr(name, data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
