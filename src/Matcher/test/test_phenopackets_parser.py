import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.DataLoader.phenopackets import (
    PhenopacketProcessor,
    ClinicalSummarizer,
    process_phenopacket
)

# A complex dummy phenopacket JSON with nested fields.
COMPLEX_PHENOPACKET = {
    "id": "complex123",
    "metaData": {
        "resources": [
            {"namespacePrefix": "HP", "version": "2021"},
            {"namespacePrefix": "MONDO"}
        ]
    },
    "subject": {
        "sex": "Female",
        "dateOfBirth": "1985-03-15",
        "timeAtLastEncounter": {"timestamp": "2022-09-15T00:00:00"},
        "taxonomy": {"label": "Homo sapiens"},
        "description": "A complex test subject with multiple features"
    },
    "phenotypicFeatures": [
        {
            "type": {"label": "HP:0001250", "description": "Seizures"},
            "excluded": False,
            "onset": {"age": {"iso8601duration": "P10Y"}},
            "modifiers": [{"label": "HP:0012301"}],
            "severity": {"label": "HP:0012823"},
            "description": "Episodes of loss of consciousness"
        },
        {
            "type": {"label": "HP:0004322", "description": "Short stature"},
            "excluded": True,
            "onset": {"timestamp": "2010-05-20T00:00:00"},
            "description": "Patient does not exhibit this feature"
        }
    ],
    "diseases": [
        {
            "term": {"label": "MONDO:0005148"},
            "diseaseStage": [{"label": "Stage II"}],
            "tnmFinding": [{"label": "T2"}],
            "onset": {"timestamp": "2021-06-01T00:00:00"},
            "description": "Primary tumor in the lung"
        }
    ],
    "biosamples": [
        {
            "sampleType": {"label": "Biopsy"},
            "sampledTissue": {"label": "Lung"},
            "timeOfCollection": {"timestamp": "2021-06-05T00:00:00"},
            "histologicalDiagnosis": {"label": "Adenocarcinoma"},
            "description": "Biopsy from lung lesion"
        }
    ],
    "medicalActions": [
        {
            "treatment": {
                "agent": {"label": "Cisplatin"},
                "routeOfAdministration": {"label": "Intravenous"},
                "doseIntervals": [{"quantity": {"value": 50, "unit": {"label": "mg/m2"}}}],
                "description": "Chemotherapy regimen"
            }
        },
        {
            "procedure": {
                "code": {"label": "Surgical Resection"},
                "performed": "2021-06-10",
                "description": "Surgical removal of tumor"
            }
        }
    ],
    "interpretations": [
        {
            "diagnosis": {
                "diagnosisStatus": {"label": "Confirmed"},
                "description": "Genomic analysis confirms mutation in TP53",
                "genomicInterpretations": [
                    {
                        "variantInterpretation": {
                            "variationDescriptor": {
                                "geneContext": {"symbol": "TP53"},
                                "label": "c.215C>G"
                            }
                        }
                    }
                ]
            },
            "description": "Overall genomic interpretation"
        }
    ],
    "measurements": [
        {
            "assay": {"id": "Assay1"},
            "value": {"value": 4.5, "unit": {"label": "mg/dL"}},
            "description": "Blood glucose level"
        }
    ],
    "family": {
        "proband": {"id": "complex123"},
        "relatives": [
            {
                "id": "relative1",
                "sex": "Male",
                "vitalStatus": {
                    "status": "Deceased",
                    "ageAtDeath": {"iso8601duration": "P70Y"}
                },
                "phenotypicFeatures": [
                    {
                        "type": {"label": "HP:0001250", "description": "Seizures"},
                        "onset": {"timestamp": "1990-01-01T00:00:00"}
                    }
                ],
                "description": "Father with history of seizures"
            }
        ],
        "pedigree": {
            "persons": [
                {"id": "complex123"},
                {"id": "relative1"}
            ]
        }
    }
}

# Dummy summary to be used with ClinicalSummarizer and process_phenopacket.
DUMMY_SUMMARY = {
    "main_conditions": ["TestCondition"],
    "other_conditions": ["AdditionalFactor"],
    "expanded_sentences": ["Expanded note 1.", "Expanded note 2."]
}


# ----- Dummy model pipeline (Mock) -----
class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # For testing, join all message content into a single string.
        return " ".join(msg["content"] for msg in messages)


class DummyPipeline:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def __call__(self, prompt, max_new_tokens, do_sample, return_full_text):
        # Return a dummy LLM output containing valid JSON.
        return [{
            "generated_text": json.dumps(DUMMY_SUMMARY)
        }]


# ----- Unit Tests -----
class TestPhenopacketProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary file to hold a complex phenopacket.
        self.temp_complex_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        json.dump(COMPLEX_PHENOPACKET, self.temp_complex_file)
        self.temp_complex_file.close()

        # Create a temporary file for an invalid phenopacket (missing required field "id").
        self.temp_invalid_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        invalid_packet = {
            "metaData": {"resources": []},
            "subject": {"sex": "Female", "dateOfBirth": "1990-01-01"}
        }
        json.dump(invalid_packet, self.temp_invalid_file)
        self.temp_invalid_file.close()

    def tearDown(self):
        os.unlink(self.temp_complex_file.name)
        os.unlink(self.temp_invalid_file.name)

    def test_complex_phenopacket_narrative_generation(self):
        """Test that a complex phenopacket is parsed and all sections are included in the narrative."""
        processor = PhenopacketProcessor(self.temp_complex_file.name)
        narrative = processor.generate_medical_narrative()
        self.assertIsInstance(narrative, list)
        self.assertGreater(len(narrative), 0)
        # Check for multiple expected sections:
        expected_sections = [
            "DEMOGRAPHICS:",
            "PHENOTYPE:",
            "DIAGNOSIS:",
            "BIOSAMPLE:",
            "TREATMENT:",
            "PROCEDURE:",
            "INTERPRETATION:",
            "MEASUREMENT:",
            "FAMILY_HISTORY:",
            "FAMILY_PEDIGREE:"
        ]
        for section in expected_sections:
            with self.subTest(section=section):
                self.assertTrue(any(section in sentence for sentence in narrative),
                                f"Section {section} not found in narrative.")

    def test_invalid_phenopacket_loading(self):
        """Test that a phenopacket missing required fields raises a ValueError."""
        with self.assertRaises(ValueError):
            _ = PhenopacketProcessor(self.temp_invalid_file.name)


class TestClinicalSummarizer(unittest.TestCase):
    @patch("src.DataLoader.phenopackets.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
    @patch("src.DataLoader.phenopackets.pipeline", return_value=DummyPipeline())
    def test_generate_summary(self, mock_pipeline, mock_tokenizer):
        """Test that ClinicalSummarizer uses the LLM pipeline correctly to generate a summary."""
        summarizer = ClinicalSummarizer(model_name="dummy-model")
        # Provide dummy narrative sentences.
        dummy_sentences = ["DEMOGRAPHICS: Female; DOB: 1985-03-15"]
        result = summarizer.generate_summary(dummy_sentences)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, DUMMY_SUMMARY)

    @patch("src.DataLoader.phenopackets.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
    @patch("src.DataLoader.phenopackets.pipeline", return_value=DummyPipeline())
    def test_extract_llm_output_invalid_json(self, mock_pipeline, mock_tokenizer):
        """Test LLM output extraction when invalid JSON is returned."""
        summarizer = ClinicalSummarizer(model_name="dummy-model")
        invalid_output = "No JSON object here"
        result = summarizer._extract_llm_output(invalid_output)
        self.assertIn("error", result)
        self.assertIn("No JSON object found", result.get("error", ""))


class TestProcessPhenopacket(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for a complex phenopacket.
        self.temp_input = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        json.dump(COMPLEX_PHENOPACKET, self.temp_input)
        self.temp_input.close()
        # Create a temporary file for output.
        self.temp_output = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        self.temp_output.close()

    def tearDown(self):
        os.unlink(self.temp_input.name)
        os.unlink(self.temp_output.name)

    @patch("src.DataLoader.phenopackets.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
    @patch("src.DataLoader.phenopackets.pipeline", return_value=DummyPipeline())
    @patch.object(ClinicalSummarizer, "generate_summary", return_value=DUMMY_SUMMARY)
    def test_process_phenopacket_end_to_end(self, mock_generate_summary, mock_pipeline, mock_tokenizer):
        """Test the end-to-end processing pipeline with patched LLM summary."""
        result = process_phenopacket(self.temp_input.name, self.temp_output.name, model="dummy-model")
        self.assertTrue(result)
        # Verify that the output file contains the dummy summary.
        with open(self.temp_output.name, "r") as f:
            output_data = json.load(f)
        self.assertEqual(output_data, DUMMY_SUMMARY)


if __name__ == "__main__":
    unittest.main()
