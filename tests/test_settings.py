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

    def test_multigpu_and_kv_knobs_survive_validation(self) -> None:
        # These fields must be DECLARED on the settings models, or the validate->model_dump
        # round-trip silently drops them (which made 0.3.2's reranker tensor_parallel_size a
        # no-op and would strip the fp8 kv_cache_dtype / max_num_seqs before they reach vLLM).
        from trialmatchai.config.settings import LLMRerankerSettings, VllmSettings

        v = VllmSettings.model_validate(
            {"kv_cache_dtype": "fp8", "max_num_seqs": 16, "tensor_parallel_size": 2}
        )
        self.assertEqual(v.model_dump()["kv_cache_dtype"], "fp8")
        self.assertEqual(v.model_dump()["max_num_seqs"], 16)
        r = LLMRerankerSettings.model_validate({"tensor_parallel_size": 2})
        self.assertEqual(r.model_dump()["tensor_parallel_size"], 2)

    def test_env_overrides(self) -> None:
        raw = {
            "search_backend": {
                "backend": "lancedb",
                "db_path": "old-search",
                "trials_table": "old-trials",
                "criteria_table": "old-criteria",
            },
            "embedder": {"model_name": "old"},
            "entity_extraction": {"backend": "gliner2"},
            "concept_linker": {"db_path": "old"},
            "patient_inputs": {"profile_dir": "old-profiles"},
            "registry": {"since_days": 7, "raw_dir": "old-raw"},
            "constraints": {"enabled": True, "score_weight": 0.25},
            "search": {"mode": "hybrid", "first_level": {"max_trials": 1000}},
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
        os.environ["TRIALMATCHAI_CONSTRAINTS_ENABLED"] = "false"
        os.environ["TRIALMATCHAI_CONSTRAINTS_SCORE_WEIGHT"] = "0.4"
        os.environ["TRIALMATCHAI_FIRST_LEVEL_ENABLED"] = "false"
        os.environ["TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS"] = "700"
        os.environ["TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE"] = "250"
        os.environ["TRIALMATCHAI_FIRST_LEVEL_VECTOR_SCORE_THRESHOLD"] = "0.1"
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
            os.environ.pop("TRIALMATCHAI_CONSTRAINTS_ENABLED", None)
            os.environ.pop("TRIALMATCHAI_CONSTRAINTS_SCORE_WEIGHT", None)
            os.environ.pop("TRIALMATCHAI_FIRST_LEVEL_ENABLED", None)
            os.environ.pop("TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS", None)
            os.environ.pop("TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE", None)
            os.environ.pop("TRIALMATCHAI_FIRST_LEVEL_VECTOR_SCORE_THRESHOLD", None)

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
        self.assertFalse(updated["constraints"]["enabled"])
        self.assertEqual(updated["constraints"]["score_weight"], 0.4)
        self.assertFalse(updated["search"]["first_level"]["enabled"])
        self.assertEqual(updated["search"]["first_level"]["max_trials"], 700)
        self.assertEqual(updated["search"]["first_level"]["per_channel_size"], 250)
        self.assertEqual(updated["search"]["first_level"]["vector_score_threshold"], 0.1)


if __name__ == "__main__":
    unittest.main()
