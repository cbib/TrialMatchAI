import os
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import xml.etree.ElementTree as ET
from download import download_study_info, get_cancer_trials_list


class GetCancerTrialsListTestCase(unittest.TestCase):
    @patch('requests.get')
    def test_get_cancer_trials_list(self, mock_get):
        # Mock the response from the requests.get function
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "FullStudiesResponse": {
                "FullStudies": [
                    {
                        "Study": {
                            "ProtocolSection": {
                                "IdentificationModule": {
                                    "NCTId": "NCT12345678"
                                }
                            }
                        }
                    },
                    {
                        "Study": {
                            "ProtocolSection": {
                                "IdentificationModule": {
                                    "NCTId": "NCT87654321"
                                }
                            }
                        }
                    }
                ]
            }
        }

        # Call the function and get the result
        result = get_cancer_trials_list(max_trials=2)

        # Assert that the requests.get function was called with the correct URL and parameters
        mock_get.assert_called_once_with(
            "https://clinicaltrials.gov/api/query/full_studies",
            params={
                "expr": "((cancer) OR (neoplasm)) AND ((interventional) OR (treatment)) AND ((mutation) OR (variant))",
                "min_rnk": 1,
                "max_rnk": 100,
                "fmt": "json",
                "fields": "NCTId"
            }
        )

        # Assert that the function returns the correct list of NCT IDs
        self.assertEqual(sorted(result), sorted(["NCT12345678", "NCT87654321"]))



class DownloadStudyInfoTestCase(unittest.TestCase):
    @patch('os.path.exists')
    @patch('requests.get')
    def test_download_study_info(self, mock_get, mock_exists):
        # Mock the response from the requests.get function
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.text = "<root><eligibility>Age less than 18</eligibility></root>"

        # Mock the os.path.exists function to return False
        mock_exists.return_value = False

        # Create a mock file object that can track what was written to it
        mock_file = mock_open()
        with patch('builtins.open', mock_file, create=True):
            # Call the function and get the result
            result = download_study_info("NCT000000000")

        # Assert that the requests.get function was called with the correct URL
        mock_get.assert_called_once_with("https://clinicaltrials.gov/ct2/show/NCT000000000?displayxml=true")

        # Assert that the function returns an empty list
        self.assertEqual(result, [])

        # Assert that the file was written with the new text
        mock_file().write.assert_called_once_with("<root><eligibility>Age less than 18</eligibility></root>")
        
    @patch('os.path.exists')
    @patch('requests.get')
    def test_download_study_info_updates_file(self, mock_get, mock_exists):
        # Mock the response from the requests.get function
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.text = "<root><eligibility>Age more than 18</eligibility></root>"

        # Mock the os.path.exists function to return True
        mock_exists.return_value = True

        # Create a dictionary to store mock file objects for each file path
        mock_files = {}

        def side_effect(file_path, mode):
            # If a mock file object for this file path does not exist, create one
            if file_path not in mock_files:
                mock_files[file_path] = mock_open(read_data="<root><eligibility>Age less than 18</eligibility></root>")()
            return mock_files[file_path]

        with patch('builtins.open', side_effect=side_effect):
            # Call the function
            download_study_info("NCT000000000")

        # Assert that the file was written with the new text
        mock_files[f"../data/trials_xmls/NCT000000000.xml"].write.assert_called_once_with("<root><eligibility>Age more than 18</eligibility></root>")
        
if __name__ == '__main__':
    unittest.main()