import unittest
from unittest.mock import patch, mock_open, MagicMock
import xml.etree.ElementTree as ET
import preprocessing_utils

class TestExtractEligibilityCriteria(unittest.TestCase):
    @patch('os.path.exists')
    @patch('xml.etree.ElementTree.ElementTree')
    @patch('xml.etree.ElementTree.fromstring')
    @patch('builtins.open', new_callable=mock_open, read_data='<xml><eligibility><criteria><textblock>Test Criteria</textblock></criteria></eligibility></xml>')
    def test_extract_eligibility_criteria(self, mock_file, mock_fromstring, mock_tree, mock_exists):
        mock_exists.return_value = True
        mock_tree_instance = mock_tree.return_value
        mock_tree_instance.getroot.return_value.find.return_value.text.strip.return_value = 'Test Criteria'
        
        mock_trial_id = MagicMock()
        mock_trial_id.__str__.return_value = 'test_trial_id'
        
        result = preprocessing_utils.extract_eligibility_criteria(mock_trial_id)
        self.assertEqual(result, 'Test Criteria')

    @patch('os.path.exists')
    def test_extract_eligibility_criteria_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        result = preprocessing_utils.extract_eligibility_criteria('test_trial_id')
        self.assertIsNone(result)

    @patch('os.path.exists')
    @patch('xml.etree.ElementTree.fromstring')
    @patch('builtins.open', new_callable=mock_open, read_data='<xml><eligibility><criteria><textblock>Test Criteria</textblock></criteria></eligibility></xml>')
    def test_extract_eligibility_criteria_parse_error(self, mock_file, mock_fromstring, mock_exists):
        mock_exists.return_value = True
        mock_fromstring.side_effect = ET.ParseError()
        
        result = preprocessing_utils.extract_eligibility_criteria('test_trial_id')
        self.assertIsNone(result)
        
class TestSplitLineToSentences(unittest.TestCase):
    @patch('preprocessing_utils.utils.replace_parentheses_with_braces')
    def test_split_line_to_sentences(self, mock_replace):
        mock_replace.return_value = '01.02.03.04 Some text here'
        regex_patterns = ['^\\d{1,2}\\.\\d{1,2}\\.\\d{1,2}\\.\\d{1,2}']
        exception_patterns = {}
        
        result = preprocessing_utils.split_line_to_sentences_by_leading_char_from_regex_patterns('01.02.03.04 Some text here', regex_patterns, exception_patterns)
        self.assertEqual(result, ['01.02.03.04 Some text here'])

    @patch('preprocessing_utils.utils.replace_parentheses_with_braces')
    def test_split_line_to_sentences_with_exception(self, mock_replace):
        mock_replace.return_value = '01.02.03 Some text here'
        regex_patterns = ['^\\d{1,2}\\.\\d{1,2}\\.\\d{1,2}']
        exception_patterns = {'03 Some': ''}
        
        result = preprocessing_utils.split_line_to_sentences_by_leading_char_from_regex_patterns('01.02.03 Some text here', regex_patterns, exception_patterns)
        self.assertEqual(result, ['01.02.03 Some text here'])

    @patch('preprocessing_utils.utils.replace_parentheses_with_braces')
    def test_split_line_to_sentences_no_match(self, mock_replace):
        mock_replace.return_value = 'No matching pattern here'
        regex_patterns = ['^\\d{1,2}\\.\\d{1,2}\\.\\d{1,2}']
        exception_patterns = {}
        
        result = preprocessing_utils.split_line_to_sentences_by_leading_char_from_regex_patterns('No matching pattern here', regex_patterns, exception_patterns)
        self.assertEqual(result, ['No matching pattern here'])

if __name__ == '__main__':
    unittest.main()