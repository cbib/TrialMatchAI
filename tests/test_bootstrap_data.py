from __future__ import annotations

import hashlib
import io
import json
import os
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
    monkeypatch.setattr("requests.get", lambda *a, **kw: _Response(b"still corrupt"))
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


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.content

    def close(self):
        pass


def _snapshot(root, version):
    data = root / "data"
    data.mkdir(exist_ok=True)
    _write_tar_gz(data / PROCESSED_TRIALS_ARCHIVE, {f"processed_trials/{version}.json": b"{}"})
    _write_zip(data / "criteria_part_0.zip", {f"{version}.json": "{}"})
    models = {f"adapter/{version}.bin": version.encode()}
    if version == "old":
        models["retired/config.json"] = b"{}"
    _write_tar_gz(data / MODELS_ARCHIVE, models)
    _write_zip(data / "finetuning_datasets.zip", {f"{version}.jsonl": "{}"})
    names = [PROCESSED_TRIALS_ARCHIVE, "criteria_part_0.zip", MODELS_ARCHIVE, "finetuning_datasets.zip"]
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(f"{_sha256(data / name)}  {name}" for name in names))
    return manifest


def _bootstrap_snapshot(root, manifest, **kwargs):
    bootstrap_data(root=root, criteria_chunks=1, with_models=True, finetune_data=True, checksum_manifest=manifest, **kwargs)


def test_changed_snapshot_removes_stale_managed_files_and_preserves_user_models(tmp_path):
    user_model = tmp_path / "models/user-model/weights.bin"
    user_model.parent.mkdir(parents=True)
    user_model.write_bytes(b"independent model")
    manifest = _snapshot(tmp_path, "old")
    unrelated = tmp_path / "data/user-notes.txt"
    unrelated.write_text("preserve")
    _bootstrap_snapshot(tmp_path, manifest)
    _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "new"))

    for folder, suffix in (("data/processed_trials", "json"), ("data/processed_criteria", "json"),
                           ("models/adapter", "bin"), ("data/finetune", "jsonl")):
        assert not (tmp_path / folder / f"old.{suffix}").exists()
        assert (tmp_path / folder / f"new.{suffix}").exists()
    assert not (tmp_path / "models/retired").exists()
    assert user_model.read_bytes() == b"independent model"
    assert unrelated.read_text() == "preserve"
    assert list((tmp_path / "data").glob(".processed_trials-backup-*/old.json"))
    marker = json.loads((tmp_path / "models/.bootstrap_complete").read_text())
    assert marker["managed_roots"] == ["adapter"]


def test_failed_extraction_keeps_previous_active_snapshot(tmp_path, monkeypatch):
    import trialmatchai.cli.bootstrap_data as module

    _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "old"))
    marker = tmp_path / "data/processed_criteria/.bootstrap_complete"
    previous_marker = marker.read_bytes()

    def interrupted(archive, target):
        (target / "partial.json").write_text("{}")
        raise OSError("simulated extraction failure")

    monkeypatch.setattr(module, "_safe_extract_zip", interrupted)
    with pytest.raises(OSError, match="simulated extraction failure"):
        _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "new"))
    assert (marker.parent / "old.json").exists()
    assert not (marker.parent / "partial.json").exists()
    assert marker.read_bytes() == previous_marker


def test_failed_publication_restores_previous_snapshot(tmp_path, monkeypatch):
    _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "old"))
    target = tmp_path / "data/processed_criteria"
    original = Path.replace

    def interrupted(source, destination):
        if source.name == "payload" and Path(destination) == target:
            raise OSError("simulated publication failure")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="simulated publication failure"):
        _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "new"))
    assert (target / "old.json").exists()
    assert not (target / "new.json").exists()


def test_next_bootstrap_recovers_an_interrupted_directory_swap(tmp_path, monkeypatch):
    _bootstrap_snapshot(tmp_path, _snapshot(tmp_path, "old"))
    target = tmp_path / "data/processed_criteria"
    target.rename(target.with_name(".processed_criteria.bootstrap-previous"))
    monkeypatch.setattr("requests.get", lambda *a, **kw: pytest.fail("valid recovered snapshot must resume"))
    _bootstrap_snapshot(tmp_path, tmp_path / "SHA256SUMS")
    assert (target / "old.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_completion_rejects_nonregular_markers(tmp_path, kind):
    from trialmatchai.cli.bootstrap_data import _extract_complete

    marker = tmp_path / ".bootstrap_complete"
    if kind == "symlink":
        external = tmp_path / "external.json"
        external.write_text(json.dumps({"schema": 2, "sha256": {}, "managed_roots": ["external.json"]}))
        marker.symlink_to(external)
    elif kind == "fifo":
        os.mkfifo(marker)
    else:
        marker.mkdir()
    assert not _extract_complete(tmp_path)
    assert not _extract_complete(tmp_path, expected_checksums={})


def test_corrupt_cached_archive_is_quarantined_and_downloaded_once(tmp_path, monkeypatch):
    from trialmatchai.cli.bootstrap_data import _download_verified

    path = tmp_path / "archive.zip"
    path.write_bytes(b"good archive")
    expected = _sha256(path)
    path.write_bytes(b"bad cache")
    calls = []
    monkeypatch.setattr("requests.get", lambda *a, **kw: calls.append(a) or _Response(b"good archive"))
    _download_verified("https://example.invalid/archive.zip", path, expected)
    assert path.read_bytes() == b"good archive"
    assert len(calls) == 1
    assert next(tmp_path.glob(".archive.zip.corrupt-*")).read_bytes() == b"bad cache"


def test_corrupt_source_has_a_bounded_retry_and_never_accepts_bad_data(tmp_path, monkeypatch):
    from trialmatchai.cli.bootstrap_data import _download_verified

    calls = []
    monkeypatch.setattr("requests.get", lambda *a, **kw: calls.append(a) or _Response(b"bad source"))
    path = tmp_path / "archive.zip"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        _download_verified("https://example.invalid/archive.zip", path, "0" * 64)
    assert len(calls) == 2
    assert not path.exists()


def test_concurrent_bootstrap_is_rejected(tmp_path):
    from trialmatchai.cli.bootstrap_data import _bootstrap_lock

    with _bootstrap_lock(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            bootstrap_data(root=tmp_path, criteria_chunks=1)
