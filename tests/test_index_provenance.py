"""Index embedder-provenance sidecar + auto-reembed on swap (Phase 4)."""

import json

from trialmatchai import orchestration as orch


def _mk_corpus(tmp_path):
    td = tmp_path / "pt"
    td.mkdir()
    (td / "N1.json").write_text(json.dumps({"nct_id": "N1", "brief_title": "t", "condition": "c"}))
    cd = tmp_path / "pc"
    (cd / "N1").mkdir(parents=True)  # criteria are grouped in per-NCT subdirectories
    (cd / "N1" / "c1.json").write_text(json.dumps({"nct_id": "N1", "criterion": "c", "criterion_type": "inclusion"}))
    return td, cd


class _FakeBackend:
    def __init__(self, db_path):
        self.db_path = db_path
        self.built = False

    def table_exists(self, name):
        return self.built

    def index_trials(self, docs, recreate=True):
        self.built = True
        return len(list(docs))

    def index_criteria(self, docs, recreate=True):
        return len(list(docs))


def _fake(tmp_path, monkeypatch):
    (tmp_path / "search").mkdir()
    fake = _FakeBackend(tmp_path / "search")
    monkeypatch.setattr(orch, "build_search_backend", lambda config: fake)
    return fake


def test_build_writes_sidecar_then_skips_on_match(tmp_path, monkeypatch):
    td, cd = _mk_corpus(tmp_path)
    fake = _fake(tmp_path, monkeypatch)
    monkeypatch.setattr(orch, "_reembed_docs_inplace", lambda *a, **k: None)
    cfg = {"embedder": {"model_name": "BAAI/bge-m3", "normalize": True}, "search_backend": {}}

    r1 = orch.build_index(cfg, processed_trials_folder=td, processed_criteria_folder=cd)
    assert r1["skipped"] is False
    assert (fake.db_path / "_embedder.json").exists()

    r2 = orch.build_index(cfg, processed_trials_folder=td, processed_criteria_folder=cd)
    assert r2["skipped"] is True  # identity matches -> fast path


def test_build_auto_reembeds_on_embedder_swap(tmp_path, monkeypatch):
    td, cd = _mk_corpus(tmp_path)
    fake = _fake(tmp_path, monkeypatch)
    reembed_calls = []
    monkeypatch.setattr(orch, "_reembed_docs_inplace", lambda *a, **k: reembed_calls.append(1))

    orch.build_index(
        {"embedder": {"model_name": "BAAI/bge-m3", "normalize": True}, "search_backend": {}},
        processed_trials_folder=td, processed_criteria_folder=cd,
    )
    # Swap the embedder: must NOT skip and must re-embed automatically (no reembed_index flag set).
    r = orch.build_index(
        {"embedder": {"model_name": "ncbi/MedCPT-Article-Encoder",
                      "query_model_name": "ncbi/MedCPT-Query-Encoder", "normalize": False},
         "search_backend": {}},
        processed_trials_folder=td, processed_criteria_folder=cd,
    )
    assert r["skipped"] is False
    assert reembed_calls == [1]
    sidecar = json.loads((fake.db_path / "_embedder.json").read_text())
    assert sidecar["identity"]["model_name"] == "ncbi/MedCPT-Article-Encoder"


def test_no_sidecar_trusts_existing_index(tmp_path, monkeypatch):
    fake = _fake(tmp_path, monkeypatch)
    fake.built = True  # tables exist, but no provenance sidecar (legacy index)
    r = orch.build_index(
        {"embedder": {"model_name": "anything"}, "search_backend": {}},
        processed_trials_folder=tmp_path / "pt", processed_criteria_folder=tmp_path / "pc",
    )
    assert r["skipped"] is True  # trusted despite unknown provenance
