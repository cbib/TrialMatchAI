from __future__ import annotations

import json

import pytest

from trialmatchai.cli.artifacts import main
from trialmatchai.utils.integrity import read_manifest, verify_manifest, write_manifest


def test_manifest_round_trip_and_corruption(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"synthetic weights\x00\xff")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/data file.json").write_text('{"synthetic": true}')
    manifest = write_manifest(tmp_path)
    assert verify_manifest(manifest, require_exact=True) == ["model.bin", "nested/data file.json"]
    (tmp_path / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_manifest(manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a/./b", "a//b", r"a\b", "C:/secret"])
def test_manifest_rejects_unsafe_paths(tmp_path, name):
    path = tmp_path / "SHA256SUMS"
    path.write_text(f"{'0' * 64}  {name}\n")
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        read_manifest(path)


def test_manifest_rejects_duplicate_and_invalid_entries(tmp_path):
    path = tmp_path / "SHA256SUMS"
    path.write_text(f"{'0' * 64}  weights.bin\n{'1' * 64}  weights.bin\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)
    path.write_text("not-a-checksum  weights.bin\n")
    with pytest.raises(ValueError, match="line 1"):
        read_manifest(path)


def test_exact_manifest_rejects_unlisted_files(tmp_path):
    (tmp_path / "package.whl").write_text("wheel")
    manifest = write_manifest(tmp_path)
    (tmp_path / "unexpected.whl").write_text("other wheel")
    with pytest.raises(ValueError, match="Unlisted artifacts"):
        verify_manifest(manifest, require_exact=True)


def test_verification_rejects_symlink_substitution(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_text("weights")
    manifest = write_manifest(tmp_path)
    weights.rename(tmp_path / "renamed.bin")
    weights.symlink_to(tmp_path / "renamed.bin")
    with pytest.raises(ValueError, match="symlink"):
        verify_manifest(manifest)


def test_artifact_cli_json_exit_status(tmp_path, capsys):
    artifact = tmp_path / "package.whl"
    artifact.write_text("wheel")
    assert main(["manifest", str(tmp_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert main(["verify", result["manifest"], "--require-exact", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    artifact.write_text("bad wheel")
    assert main(["verify", result["manifest"], "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize("explicit_directory", [False, True])
def test_manifest_symlink_cannot_redirect_verification(tmp_path, capsys, explicit_directory):
    trusted = tmp_path / "trusted"
    incoming = tmp_path / "incoming"
    trusted.mkdir()
    incoming.mkdir()
    (trusted / "package.whl").write_bytes(b"known good")
    (incoming / "package.whl").write_bytes(b"corrupted incoming wheel")
    manifest = write_manifest(trusted)
    link = incoming / "SHA256SUMS"
    link.symlink_to(manifest)

    arguments = ["verify", str(link), "--require-exact", "--json"]
    if explicit_directory:
        arguments += ["--directory", str(incoming)]
    assert main(arguments) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "manifest must not be a symlink" in result["error"]


def test_external_manifest_verifies_the_explicit_artifact_directory(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "package.whl").write_bytes(b"wheel")
    manifest = write_manifest(artifacts).rename(tmp_path / "SHA256SUMS")
    assert verify_manifest(manifest, directory=artifacts, require_exact=True) == ["package.whl"]
