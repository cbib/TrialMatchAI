import os
import unittest
from pathlib import Path

from Matcher.config.config_loader import load_config
from Matcher.config.settings import apply_env_overrides


class TestConfigLoading(unittest.TestCase):
    def test_load_config_from_repo(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "source/Matcher/config/config.json"
        config = load_config(str(config_path))
        self.assertIn("elasticsearch", config)
        self.assertIn("embedder", config)
        self.assertIn("paths", config)

    def test_env_overrides(self) -> None:
        raw = {
            "elasticsearch": {
                "host": "http://localhost:9200",
                "username": "user",
                "password": "pass",
            },
            "embedder": {"model_name": "old"},
        }
        os.environ["TRIALMATCHAI_ES_HOST"] = "http://override:9200"
        os.environ["TRIALMATCHAI_EMBEDDER_MODEL_NAME"] = "new-model"
        os.environ["TRIALMATCHAI_ES_AUTO_START"] = "true"
        try:
            updated = apply_env_overrides(raw)
        finally:
            os.environ.pop("TRIALMATCHAI_ES_HOST", None)
            os.environ.pop("TRIALMATCHAI_EMBEDDER_MODEL_NAME", None)
            os.environ.pop("TRIALMATCHAI_ES_AUTO_START", None)

        self.assertEqual(updated["elasticsearch"]["host"], "http://override:9200")
        self.assertEqual(updated["embedder"]["model_name"], "new-model")
        self.assertTrue(updated["elasticsearch"]["auto_start"])


if __name__ == "__main__":
    unittest.main()
