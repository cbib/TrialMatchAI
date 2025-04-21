"""
test_preprocessing.py

This file contains unit tests for the preprocessing functions used for handling
clinical trial eligibility criteria texts.
"""

import os
import re
import json
import tempfile
import unittest
import itertools
import pandas as pd
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock


from src.Preprocessor.preprocessing_utils import (
    load_regex_patterns,
    split_on_leading_markers,
    replace_parentheses_with_braces,
    replace_braces_with_parentheses,
    line_starts_with_capitalized_alphanumeric,
    read_xml_file,
    parse_xml_content,
    extract_eligibility_criteria,
    split_by_leading_char_from_regex_patterns,
    is_header,
    is_false_header,
    split_on_carriage_returns,
    split_lines_on_semicolon,
    split_to_sentences,
    drop_leading_character,
    extract_criteria_sections_headers,
    fix_inline_headers,
    extract_separate_inclusion_exclusion,
    split_on_full_stops,
    split_large_sentences,
    eic_text_preprocessing
)


class TestLoadRegexPatterns(unittest.TestCase):
    def setUp(self):
        # Create a temporary JSON file with a patterns dictionary.
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        sample_data = {
            "patterns": {
                "bullet": {"regex": r"•"},
                "dash": {"regex": r"^-"},
                "number": {"regex": r"^\d+\)"}
            }
        }
        json.dump(sample_data, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_load_regex_patterns(self):
        patterns = load_regex_patterns(self.temp_file.name)
        expected = {
            "bullet": r"•",
            "dash": r"^-",
            "number": r"^\d+\)"
        }
        self.assertEqual(patterns, expected)


class TestSplittingAndReplacingFunctions(unittest.TestCase):
    def test_split_on_leading_markers(self):
        # Test splitting a line with bullet, dash and asterisk markers.
        input_lines = [
            "• First item - subitem * detail",
            "No marker here"
        ]
        result = split_on_leading_markers(input_lines)
        # Check that more than one line is returned and that it contains expected fragments.
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)
        self.assertIn("First item", result[0])
        self.assertIn("subitem", " ".join(result))

    def test_replace_parentheses_with_braces(self):
        input_text = "This is a test (example) with [brackets]."
        expected = "This is a test {example} with {brackets}."
        # Note: The implementation replaces both '(' and '[' with '{' and their closing pairs with '}'
        result = replace_parentheses_with_braces(input_text)
        self.assertEqual(result, expected)

    def test_replace_braces_with_parentheses(self):
        input_text = "This is a test {example} with {brackets}."
        expected = "This is a test (example) with (brackets)."
        result = replace_braces_with_parentheses(input_text)
        self.assertEqual(result, expected)

    def test_line_starts_with_capitalized_alphanumeric(self):
        self.assertTrue(line_starts_with_capitalized_alphanumeric("Hello world"))
        self.assertFalse(line_starts_with_capitalized_alphanumeric("hello world"))
        self.assertFalse(line_starts_with_capitalized_alphanumeric("  "))
        self.assertTrue(line_starts_with_capitalized_alphanumeric("A1 is valid"))

    def test_split_by_leading_char_from_regex_patterns(self):
        # Use a regex that matches any numeric marker in the text (without the start-of-string anchor).
        line = "1) First sentence. 2) Second sentence."
        patterns = [r"\b\d+\)"]
        result = split_by_leading_char_from_regex_patterns(line, patterns)
        # Expect two parts: one starting from "1)" and another starting from "2)"
        self.assertGreaterEqual(len(result), 2, f"Result was {result}")


class TestXMLFunctions(unittest.TestCase):
    def setUp(self):
        # Create temporary XML content.
        self.valid_xml = "<root><eligibility><criteria><textblock>Eligibility Content</textblock></criteria></eligibility></root>"
        self.invalid_xml = "<root><eligibility><criteria><textblock>Missing closing tags"
        # Create a temporary file for testing read_xml_file.
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml")
        self.temp_xml_file.write(self.valid_xml)
        self.temp_xml_file.close()

    def tearDown(self):
        os.unlink(self.temp_xml_file.name)

    def test_read_xml_file_success(self):
        content = read_xml_file(self.temp_xml_file.name)
        self.assertIn("Eligibility Content", content)

    def test_read_xml_file_failure(self):
        # Attempt to read a non-existent file.
        content = read_xml_file("nonexistent_file.xml")
        self.assertIsNone(content)

    def test_parse_xml_content_success(self):
        root = parse_xml_content(self.valid_xml)
        self.assertIsNotNone(root)
        self.assertEqual(root.tag, "root")

    def test_parse_xml_content_failure(self):
        root = parse_xml_content(self.invalid_xml)
        self.assertIsNone(root)


class TestExtractEligibilityCriteria(unittest.TestCase):
    def setUp(self):
        # Create dummy XML content that includes the eligibility textblock.
        self.dummy_xml = "<root><eligibility><criteria><textblock>Eligibility Content for Trial</textblock></criteria></eligibility></root>"
        # Create a temporary file that will be used to simulate a trial XML file.
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml")
        self.temp_xml_file.write(self.dummy_xml)
        self.temp_xml_file.close()

    def tearDown(self):
        os.unlink(self.temp_xml_file.name)

    def test_extract_eligibility_criteria_found(self):
        trial_id = "dummy_trial"
        # Patch os.path.exists and os.path.join using the correct module path.
        with patch("src.Preprocessor.preprocessing_utils.os.path.exists", return_value=True), \
             patch("src.Preprocessor.preprocessing_utils.os.path.join", return_value=self.temp_xml_file.name):
            text = extract_eligibility_criteria(trial_id)
            self.assertEqual(text, "Eligibility Content for Trial")

    def test_extract_eligibility_criteria_not_found(self):
        trial_id = "dummy_trial"
        # Return False for file existence.
        with patch("src.Preprocessor.preprocessing_utils.os.path.exists", return_value=False):
            text = extract_eligibility_criteria(trial_id)
            self.assertIsNone(text)


class TestOtherTextProcessingFunctions(unittest.TestCase):
    def test_split_on_carriage_returns(self):
        text = "Line one.\n\nLine two.\n\n\nLine three."
        result = split_on_carriage_returns(text)
        self.assertEqual(len(result), 3)
        self.assertIn("Line two.", result)

    def test_split_lines_on_semicolon(self):
        lines = ["Sentence one; Sentence two {ignore; this} end."]
        result = split_lines_on_semicolon(lines)
        # Expect the semicolon outside braces to split the line
        # and preserve the entire text "ignore; this" inside braces.
        self.assertIn("Sentence one", result[0])
        self.assertEqual(result[1], "Sentence two {ignore; this} end.")

    def test_split_to_sentences(self):
        text = "Inclusion Criteria: Patients over 18. Exclusion Criteria: Non-eligible subjects."
        regex_patterns = [r"Inclusion Criteria", r"Exclusion Criteria"]
        exception_patterns = []
        result = split_to_sentences(text, regex_patterns, exception_patterns)
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    def test_drop_leading_character(self):
        # Suppose our regex patterns remove leading numbers and punctuation.
        patterns = [r"^\d+\.", r"^-"]
        sentence = "1. This is a test sentence."
        result = drop_leading_character(sentence, patterns)
        self.assertNotEqual(result, sentence)
        self.assertTrue(result.startswith("This"))

    def test_extract_criteria_sections_headers(self):
        # Provide a list of simulated lines where some lines look like headers.
        lines = [
            "Inclusion Criteria:",
            "Patients must be over 18.",
            "Exclusion Criteria:",
            "Patients with comorbidities."
        ]
        sections = extract_criteria_sections_headers(lines)
        self.assertIsInstance(sections, dict)
        # There should be keys that contain "Inclusion" and "Exclusion".
        keys_joined = " ".join(sections.keys()).lower()
        self.assertIn("inclusion", keys_joined)
        self.assertIn("exclusion", keys_joined)

    def test_fix_inline_headers(self):
        text = "Inclusion Criteria:Patients must be over 18. Exclusion Criteria:Patients with comorbidities."
        fixed = fix_inline_headers(text)
        # Check that there is a newline inserted after the colon for each header.
        self.assertIn("Criteria:\n", fixed)

    def test_extract_separate_inclusion_exclusion(self):
        # Create a dummy eligibility text with inline headers.
        text = "Inclusion Criteria: Patients must be over 18. Exclusion Criteria: Patients with comorbidities."
        regex_patterns = [r"Inclusion Criteria", r"Exclusion Criteria"]
        exception_patterns = []
        result = extract_separate_inclusion_exclusion(text, regex_patterns, exception_patterns)
        self.assertIn("Inclusion Criteria", result)
        self.assertIn("Exclusion Criteria", result)
        # The text for each header should be non-empty dictionaries.
        self.assertIsInstance(result["Inclusion Criteria"], dict)
        self.assertIsInstance(result["Exclusion Criteria"], dict)

    def test_split_on_full_stops(self):
        text = "This is a sentence. And here is another? Yes, indeed. Final sentence."
        sentences = split_on_full_stops(text)
        self.assertGreaterEqual(len(sentences), 3)
        self.assertTrue(any("Final sentence" in s for s in sentences))

    def test_split_large_sentences(self):
        # Create a DataFrame with one long sentence.
        df = pd.DataFrame({
            "sentence": ["This is a very long sentence. It should be split. Here is another sentence."],
            "criteria": ["Inclusion"],
            "sub_criteria": ["General"],
            "id": ["trial1"]
        })
        new_df = split_large_sentences(df)
        # We expect that if the threshold is 200 characters, a sentence shorter than that remains unchanged.
        self.assertFalse(new_df.empty)
        self.assertTrue(new_df["sentence"].str.len().min() > 0)


class TestEICTextPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create temporary dummy regex patterns files.
        self.temp_regex_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        regex_data = {
            "patterns": {
                "pattern1": {"regex": r"\d+\)"},
                "pattern2": {"regex": r"[•*]"}
            }
        }
        json.dump(regex_data, self.temp_regex_file)
        self.temp_regex_file.close()

        self.temp_exceptions_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        exceptions_data = {
            "patterns": {
                "exception": {"regex": r"EXCEPTION"}
            }
        }
        json.dump(exceptions_data, self.temp_exceptions_file)
        self.temp_exceptions_file.close()

        # Create a dummy eligibility criteria XML content.
        self.dummy_xml = "<root><eligibility><criteria><textblock>Inclusion Criteria: Patients over 18. Exclusion Criteria: Patients with comorbidities.</textblock></criteria></eligibility></root>"
        # Create a temporary XML file to be read by extract_eligibility_criteria.
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml")
        self.temp_xml_file.write(self.dummy_xml)
        self.temp_xml_file.close()

        # Create a temporary output directory.
        self.temp_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self.temp_regex_file.name)
        os.unlink(self.temp_exceptions_file.name)
        os.unlink(self.temp_xml_file.name)
        # Remove output files if created.
        for f in os.listdir(self.temp_output_dir):
            os.unlink(os.path.join(self.temp_output_dir, f))
        os.rmdir(self.temp_output_dir)

    def dummy_extract_eligibility_criteria(self, trial_id):
        # Instead of reading a file from disk, simply return the text inside our dummy XML.
        return "Inclusion Criteria: Patients over 18. Exclusion Criteria: Patients with comorbidities."

    @patch("src.Preprocessor.preprocessing_utils.extract_eligibility_criteria")
    def test_eic_text_preprocessing(self, mock_extract):
        # Patch extract_eligibility_criteria to return a dummy string.
        mock_extract.side_effect = self.dummy_extract_eligibility_criteria
        _ids = ["trial123"]
        df = eic_text_preprocessing(
            _ids,
            regex_path=self.temp_regex_file.name,
            exceptions_path=self.temp_exceptions_file.name,
            output_path=self.temp_output_dir
        )
        self.assertIsNotNone(df)
        # Check that the DataFrame contains expected columns.
        for col in ["sentence", "criteria", "sub_criteria", "id"]:
            self.assertIn(col, df.columns)
        # Check that an output file was written to the output directory.
        files = os.listdir(self.temp_output_dir)
        self.assertTrue(any(f.endswith("_preprocessed.tsv") for f in files))


if __name__ == '__main__':
    unittest.main()
