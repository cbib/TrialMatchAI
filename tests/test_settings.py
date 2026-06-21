import os
import unittest
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.config.settings import apply_env_overrides


class TestConfigLoading(unittest.TestCase):
    def test_load_config_from_repo(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"
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
            "patient_inputs": {"profile_dir": "old-profiles"},
            "registry": {"since_days": 7, "raw_dir": "old-raw"},
        }
        os.environ["TRIALMATCHAI_SEARCH_DB_PATH"] = "data/search-test"
        os.environ["TRIALMATCHAI_SEARCH_TRIALS_TABLE"] = "trials-test"
        os.environ["TRIALMATCHAI_SEARCH_MODE"] = "bm25"
        os.environ["TRIALMATCHAI_EMBEDDER_MODEL_NAME"] = "new-model"
        os.environ["TRIALMATCHAI_ENTITY_BACKEND"] = "regex"
        os.environ["TRIALMATCHAI_CONCEPT_DB_PATH"] = "concepts"
        os.environ["TRIALMATCHAI_PATIENT_PROFILE_DIR"] = "patients/profiles"
        os.environ["TRIALMATCHAI_PATIENT_STRICT_VALIDATION"] = "true"
        os.environ["TRIALMATCHAI_REGISTRY_SINCE_DAYS"] = "30"
        os.environ["TRIALMATCHAI_REGISTRY_RAW_DIR"] = "registry/raw"
        try:
            updated = apply_env_overrides(raw)
        finally:
            os.environ.pop("TRIALMATCHAI_SEARCH_DB_PATH", None)
            os.environ.pop("TRIALMATCHAI_SEARCH_TRIALS_TABLE", None)
            os.environ.pop("TRIALMATCHAI_SEARCH_MODE", None)
            os.environ.pop("TRIALMATCHAI_EMBEDDER_MODEL_NAME", None)
            os.environ.pop("TRIALMATCHAI_ENTITY_BACKEND", None)
            os.environ.pop("TRIALMATCHAI_CONCEPT_DB_PATH", None)
            os.environ.pop("TRIALMATCHAI_PATIENT_PROFILE_DIR", None)
            os.environ.pop("TRIALMATCHAI_PATIENT_STRICT_VALIDATION", None)
            os.environ.pop("TRIALMATCHAI_REGISTRY_SINCE_DAYS", None)
            os.environ.pop("TRIALMATCHAI_REGISTRY_RAW_DIR", None)

        self.assertEqual(updated["search_backend"]["db_path"], "data/search-test")
        self.assertEqual(updated["search_backend"]["trials_table"], "trials-test")
        self.assertEqual(updated["search"]["mode"], "bm25")
        self.assertEqual(updated["embedder"]["model_name"], "new-model")
        self.assertEqual(updated["entity_extraction"]["backend"], "regex")
        self.assertEqual(updated["concept_linker"]["db_path"], "concepts")
        self.assertEqual(updated["patient_inputs"]["profile_dir"], "patients/profiles")
        self.assertTrue(updated["patient_inputs"]["strict_validation"])
        self.assertEqual(updated["registry"]["since_days"], 30)
        self.assertEqual(updated["registry"]["raw_dir"], "registry/raw")


if __name__ == "__main__":
    unittest.main()
