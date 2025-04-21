import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import xml.etree.ElementTree as ET

from src.DataLoader.downloader import (
    normalize_whitespace,
    get_cancer_trials_list,
    download_study_info,
    parallel_downloader,
    Downloader,
)

class TestDownloaderFunctions(unittest.TestCase):
    def test_normalize_whitespace(self):
        input_str = "  This  is   a test string \n with   extra   spaces.  "
        expected = "This is a test string with extra spaces."
        self.assertEqual(normalize_whitespace(input_str), expected)

    @patch("src.DataLoader.downloader.requests.get")
    def test_get_cancer_trials_list(self, mock_get):
        # Set up a fake study entry
        fake_study = {
            "Study": {
                "ProtocolSection": {
                    "IdentificationModule": {"NCTId": "NCT12345678"}
                }
            }
        }
        fake_json_response = {
            "FullStudiesResponse": {"FullStudies": [fake_study]}
        }
        # Create a fake response object for the first call
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = fake_json_response

        # Second call returns an empty list to stop the loop.
        fake_response_empty = MagicMock()
        fake_response_empty.status_code = 200
        fake_response_empty.json.return_value = {"FullStudiesResponse": {"FullStudies": []}}

        mock_get.side_effect = [fake_response, fake_response_empty]

        result = get_cancer_trials_list(max_trials=1)
        self.assertIn("NCT12345678", result)

    @patch("src.DataLoader.downloader.time.sleep", return_value=None)  # avoid delay
    @patch("src.DataLoader.downloader.requests.Session")
    @patch("src.DataLoader.downloader.os.path.exists")
    @patch("src.DataLoader.downloader.os.makedirs")
    @patch("src.DataLoader.downloader.open", new_callable=mock_open, read_data="")
    def test_download_study_info_no_local_file(self, mock_file, mock_makedirs, mock_exists, mock_session_cls, mock_sleep):
        # Simulate that the local file does not exist
        mock_exists.return_value = False

        # Prepare a fake online XML
        online_xml = '''<clinical_study>
  <brief_title>Test Trial</brief_title>
  <eligibility>Test Eligibility</eligibility>
  <overall_status>Recruiting</overall_status>
  <location>Test Location</location>
</clinical_study>'''
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.text = online_xml

        # Create a fake session instance
        fake_session = MagicMock()
        fake_session.get.return_value = fake_response
        mock_session_cls.return_value = fake_session

        result = download_study_info("NCTTEST1", delay=0, session=fake_session)
        self.assertTrue(result)
        # Check that the file was opened for writing.
        expected_path = os.path.join("..", "..", "data", "trials_xmls", "NCTTEST1.xml")
        mock_file.assert_called_with(expected_path, "w", encoding="utf-8")

    @patch("src.DataLoader.downloader.time.sleep", return_value=None)
    @patch("src.DataLoader.downloader.requests.Session")
    @patch("src.DataLoader.downloader.os.path.exists")
    @patch("src.DataLoader.downloader.os.makedirs")
    def test_download_study_info_with_local_file_update_needed(self, mock_makedirs, mock_exists, mock_session_cls, mock_sleep):
        # Simulate that the local file exists.
        mock_exists.return_value = True

        # Fake local XML (with an old title) and online XML (with an updated title)
        local_xml = '''<clinical_study>
  <brief_title>Old Title</brief_title>
  <eligibility>Test Eligibility</eligibility>
  <overall_status>Recruiting</overall_status>
  <location>Test Location</location>
</clinical_study>'''
        online_xml = '''<clinical_study>
  <brief_title>New Title</brief_title>
  <eligibility>Test Eligibility</eligibility>
  <overall_status>Recruiting</overall_status>
  <location>Test Location</location>
</clinical_study>'''

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.text = online_xml

        fake_session = MagicMock()
        fake_session.get.return_value = fake_response
        mock_session_cls.return_value = fake_session

        # Patch open to simulate reading the local XML and then writing an updated version.
        m = mock_open(read_data=local_xml)
        with patch("src.DataLoader.downloader.open", m):
            result = download_study_info("NCTTEST2", delay=0, session=fake_session)
            self.assertTrue(result)
            expected_path = os.path.join("..", "..", "data", "trials_xmls", "NCTTEST2.xml")
            # Check that open was called with write mode to update the file.
            m.assert_called_with(expected_path, "w", encoding="utf-8")

    @patch("src.DataLoader.downloader.time.sleep", return_value=None)
    @patch("src.DataLoader.downloader.requests.Session")
    @patch("src.DataLoader.downloader.os.path.exists")
    @patch("src.DataLoader.downloader.os.makedirs")
    def test_download_study_info_with_local_file_no_update(self, mock_makedirs, mock_exists, mock_session_cls, mock_sleep):
        # Simulate that the local file exists.
        mock_exists.return_value = True

        # Fake XML where both local and online versions are identical.
        identical_xml = '''<clinical_study>
  <brief_title>Test Trial</brief_title>
  <eligibility>Test Eligibility</eligibility>
  <overall_status>Recruiting</overall_status>
  <location>Test Location</location>
</clinical_study>'''

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.text = identical_xml

        fake_session = MagicMock()
        fake_session.get.return_value = fake_response
        mock_session_cls.return_value = fake_session

        m = mock_open(read_data=identical_xml)
        with patch("src.DataLoader.downloader.open", m):
            result = download_study_info("NCTTEST3", delay=0, session=fake_session)
            self.assertTrue(result)
            expected_path = os.path.join("..", "..", "data", "trials_xmls", "NCTTEST3.xml")
            # Check that open was called only for reading (the file was not updated).
            m.assert_called_with(expected_path, "r", encoding="utf-8")

    @patch("src.DataLoader.downloader.download_study_info", return_value=True)
    @patch("src.DataLoader.downloader.tqdm", lambda x, **kwargs: x)  # bypass tqdm for testing
    def test_parallel_downloader(self, mock_download):
        trial_ids = ["NCT1", "NCT2", "NCT3"]
        # Use n_jobs=1 to avoid issues with multiprocessing and patching.
        results = parallel_downloader(trial_ids, n_jobs=1, delay=0)
        self.assertEqual(results, [True, True, True])
        self.assertEqual(mock_download.call_count, 3)

    @patch("src.DataLoader.downloader.parallel_downloader", return_value=[True, True])
    def test_downloader_class(self, mock_parallel):
        trial_ids = ["NCT1", "NCT2"]
        downloader_obj = Downloader(trial_ids, n_jobs=2, delay=0)
        results = downloader_obj.download_and_update_trials()
        self.assertEqual(results, [True, True])
        mock_parallel.assert_called_with(trial_ids, n_jobs=2, delay=0)

if __name__ == "__main__":
    unittest.main()
