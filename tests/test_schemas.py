import unittest

from Matcher.schemas.phenopacket import Keywords, Phenopacket


class TestSchemas(unittest.TestCase):
    def test_phenopacket_minimal(self) -> None:
        data = {"id": "patient-1", "metaData": {}, "subject": {}}
        obj = Phenopacket.model_validate(data)
        self.assertEqual(obj.id, "patient-1")

    def test_keywords_default(self) -> None:
        data = {"main_conditions": ["A"], "other_conditions": [], "expanded_sentences": []}
        obj = Keywords.model_validate(data)
        self.assertEqual(obj.main_conditions, ["A"])

    def test_keywords_allows_extra(self) -> None:
        data = {
            "main_conditions": [],
            "other_conditions": [],
            "expanded_sentences": [],
            "error": "bad",
            "extra": "ok",
        }
        obj = Keywords.model_validate(data)
        self.assertEqual(obj.error, "bad")


if __name__ == "__main__":
    unittest.main()
