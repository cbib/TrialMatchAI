"""Model catalog: one-line `model: "<name>"` swap (Phase 6)."""

import json
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.search import build_search_backend

REPO_CONFIG = Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"


def _load(tmp_path, embedder, search_backend=None):
    cfg = json.loads(REPO_CONFIG.read_text())
    cfg["embedder"] = embedder
    if search_backend is not None:
        cfg["search_backend"] = search_backend
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    return load_config(str(p))


def test_catalog_expands_medcpt_shorthand(tmp_path):
    cfg = _load(tmp_path, {"model": "medcpt"})
    e = cfg["embedder"]
    assert e["model_name"] == "ncbi/MedCPT-Article-Encoder"
    assert e["query_model_name"] == "ncbi/MedCPT-Query-Encoder"
    assert e["normalize"] is False
    assert e["native_metric"] == "dot"


def test_catalog_swap_makes_metric_follow(tmp_path):
    # Unpinned vector_metric -> follows the embedder chosen by the catalog.
    cfg = _load(tmp_path, {"model": "medcpt"})
    assert build_search_backend(cfg).vector_metric == "dot"
    cfg_bge = _load(tmp_path, {"model": "bge-m3"})
    assert build_search_backend(cfg_bge).vector_metric == "cosine"


def test_inline_keys_override_catalog(tmp_path):
    cfg = _load(tmp_path, {"model": "medcpt", "max_length": 256, "normalize": True})
    assert cfg["embedder"]["max_length"] == 256
    assert cfg["embedder"]["normalize"] is True


def test_unknown_model_falls_back_to_model_name(tmp_path):
    cfg = _load(tmp_path, {"model": "some/random-model"})
    assert cfg["embedder"]["model_name"] == "some/random-model"


def test_explicit_metric_still_pins_over_embedder(tmp_path):
    cfg = _load(tmp_path, {"model": "medcpt"}, search_backend={"vector_metric": "cosine"})
    assert build_search_backend(cfg).vector_metric == "cosine"


def test_legacy_explicit_model_name_untouched(tmp_path):
    cfg = _load(tmp_path, {"backend": "hf", "model_name": "BAAI/bge-m3", "normalize": True})
    assert cfg["embedder"]["model_name"] == "BAAI/bge-m3"
