from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.entities.schemas import default_schema_path


def test_load_config_from_repo():
    config_path = Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"
    cfg = load_config(str(config_path))
    assert cfg["search_backend"]["backend"] == "lancedb"
    assert "embedder" in cfg
    assert "paths" in cfg


def test_packaged_schema_path_resolves_outside_repo(tmp_path, monkeypatch):
    source_config = (
        Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"
    )
    installed_config = tmp_path / "site-packages/trialmatchai/config/config.json"
    installed_config.parent.mkdir(parents=True)
    installed_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = load_config(installed_config)

    assert cfg["entity_extraction"]["schema_path"] == str(default_schema_path().resolve())
    assert Path(cfg["entity_extraction"]["schema_path"]).exists()
    assert cfg["paths"]["output_dir"] == str((tmp_path / "results").resolve())
