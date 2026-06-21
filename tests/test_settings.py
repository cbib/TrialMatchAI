import os
import unittest
from pathlib import Path

from Matcher.config.config_loader import load_config
from Matcher.config.settings import apply_env_overrides


class TestConfigLoading(unittest.TestCase):
    def test_load_config_from_repo(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "source/Matcher/config/config.json"
        config = load_config(str(config_path))
        self.assertIn("search_backend", config)
        self.assertIn("embedder", config)
        self.assertIn("paths", config)

    def test_env_overrides(self) -> None:
        raw = {
            "search_backend": {
                "backend": "lancedb",
                "db_path": "old-search",
                "trials_table": "old-trials",
                "criteria_table": "old-criteria",
            },
            "embedder": {"model_name": "old"},
            "search": {"mode": "hybrid"},
            "entity_extraction": {"backend": "gliner2"},
            "concept_linker": {"db_path": "old"},
        }
        os.environ["TRIALMATCHAI_SEARCH_DB_PATH"] = "data/search-test"
        os.environ["TRIALMATCHAI_SEARCH_TRIALS_TABLE"] = "trials-test"
        os.environ["TRIALMATCHAI_SEARCH_MODE"] = "bm25"
        os.environ["TRIALMATCHAI_EMBEDDER_MODEL_NAME"] = "new-model"
        os.environ["TRIALMATCHAI_ENTITY_BACKEND"] = "regex"
        os.environ["TRIALMATCHAI_CONCEPT_DB_PATH"] = "concepts"
        try:
            updated = apply_env_overrides(raw)
        finally:
            os.environ.pop("TRIALMATCHAI_SEARCH_DB_PATH", None)
            os.environ.pop("TRIALMATCHAI_SEARCH_TRIALS_TABLE", None)
            os.environ.pop("TRIALMATCHAI_SEARCH_MODE", None)
            os.environ.pop("TRIALMATCHAI_EMBEDDER_MODEL_NAME", None)
            os.environ.pop("TRIALMATCHAI_ENTITY_BACKEND", None)
            os.environ.pop("TRIALMATCHAI_CONCEPT_DB_PATH", None)

        self.assertEqual(updated["search_backend"]["db_path"], "data/search-test")
        self.assertEqual(updated["search_backend"]["trials_table"], "trials-test")
        self.assertEqual(updated["search"]["mode"], "bm25")
        self.assertEqual(updated["embedder"]["model_name"], "new-model")
        self.assertEqual(updated["entity_extraction"]["backend"], "regex")
        self.assertEqual(updated["concept_linker"]["db_path"], "concepts")


if __name__ == "__main__":
    unittest.main()
