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
        with_models=True,
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
        with_models=True,
    )


def test_verify_sha256_rejects_mismatches(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("contents")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_sha256(path, "0" * 64)


def test_strict_bootstrap_requires_all_checksums_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("TRIALMATCHAI_PROCESSED_TRIALS_SHA256", raising=False)
    monkeypatch.delenv("TRIALMATCHAI_CRITERIA_PART_0_SHA256", raising=False)
    with pytest.raises(ValueError, match="Missing required SHA-256"):
        bootstrap_data(root=tmp_path, criteria_chunks=1, require_checksums=True)
    assert not (tmp_path / "data").exists()


def test_strict_bootstrap_verifies_cached_archives_and_resume_provenance(tmp_path, monkeypatch):
    from trialmatchai.cli.bootstrap_data import _extract_complete
    from trialmatchai.utils.integrity import write_manifest

    data = tmp_path / "data"
    data.mkdir()
    _write_tar_gz(data / PROCESSED_TRIALS_ARCHIVE, {"processed_trials/NCT00000001.json": b"{}"})
    _write_zip(data / "criteria_part_0.zip", {"c1.json": "{}"})
    manifest = write_manifest(data)
    sums = {"criteria_part_0.zip": _sha256(data / "criteria_part_0.zip")}
    bootstrap_data(root=tmp_path, criteria_chunks=1, checksum_manifest=manifest)
    assert _extract_complete(data / "processed_criteria", expected_checksums=sums)
    monkeypatch.setattr("requests.get", lambda *a, **kw: pytest.fail("Verified resume must not download"))
    bootstrap_data(root=tmp_path, criteria_chunks=1, checksum_manifest=manifest)
    assert not _extract_complete(data / "processed_criteria", expected_checksums={"criteria_part_0.zip": "0" * 64})


def test_legacy_marker_does_not_count_as_verified(tmp_path):
    from trialmatchai.cli.bootstrap_data import _extract_complete

    (tmp_path / ".bootstrap_complete").write_text("ok\n")
    assert not _extract_complete(tmp_path, expected_checksums={"archive.zip": "0" * 64})


def test_strict_bootstrap_corruption_prevents_extraction(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "criteria_part_0.zip").write_bytes(b"corrupt")
    monkeypatch.setenv("TRIALMATCHAI_PROCESSED_TRIALS_SHA256", "0" * 64)
    monkeypatch.setenv("TRIALMATCHAI_CRITERIA_PART_0_SHA256", "0" * 64)
    monkeypatch.setattr("trialmatchai.cli.bootstrap_data._safe_extract_zip", lambda *a: pytest.fail("Must not extract corrupt data"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        bootstrap_data(root=tmp_path, criteria_chunks=1, require_checksums=True)
    assert not (data / "processed_criteria/.bootstrap_complete").exists()


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
