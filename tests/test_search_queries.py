import unittest

from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch


class DummyES:
    def search(self, *args, **kwargs):
        raise AssertionError("ES search should not be called in these tests.")


class TestFirstLevelQueryBuilding(unittest.TestCase):
    def setUp(self) -> None:
        self.search = ClinicalTrialSearch(
            es_client=DummyES(),
            embedder=None,
            index_name="index",
            bio_med_ner=None,
        )

    def test_create_query_bm25(self) -> None:
        query = self.search.create_query(
            synonyms=["lung cancer"],
            embeddings={},
            age=45,
            sex="ALL",
            overall_status="Recruiting",
            max_text_score=1.0,
            vector_score_threshold=0.5,
            pre_selected_nct_ids=None,
            other_conditions=None,
            search_mode="bm25",
        )
        self.assertIn("bool", query)
        self.assertIn("should", query["bool"])
        self.assertIn("filter", query["bool"])

    def test_create_query_vector(self) -> None:
        embeddings = {"lung cancer": [0.1, 0.2], "smoking": [0.3, 0.4]}
        query = self.search.create_query(
            synonyms=["lung cancer"],
            embeddings=embeddings,
            age=60,
            sex="MALE",
            overall_status=None,
            max_text_score=1.0,
            vector_score_threshold=0.2,
            pre_selected_nct_ids=None,
            other_conditions=["smoking"],
            search_mode="vector",
        )
        self.assertIn("script_score", query)
        script = query["script_score"]["script"]
        self.assertIn("params", script)
        self.assertEqual(len(script["params"]["query_vectors"]), 1)
        self.assertEqual(len(script["params"]["other_condition_vectors"]), 1)


if __name__ == "__main__":
    unittest.main()
