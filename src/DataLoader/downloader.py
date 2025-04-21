import requests
import sys
import xml.etree.ElementTree as ET
import os
import time
import joblib
from tqdm.auto import tqdm
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pandas as pd
import logging
from typing import List, Union

# Configure logging for clear and timestamped output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in a string."""
    return ' '.join(s.split())


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def get_cancer_trials_list(max_trials: int = 15000) -> List[str]:
    """
    Retrieve a list of cancer-related clinical trial NCT IDs from ClinicalTrials.gov.
    
    Args:
        max_trials (int): Maximum number of trial IDs to fetch.
    
    Returns:
        List[str]: List of unique NCT IDs.
    """
    base_url = "https://clinicaltrials.gov/api/query/full_studies"
    trials_set = set()
    page_size = 100  # Number of trials per page
    current_rank = 1
    trials_fetched = 0

    while trials_fetched < max_trials:
        search_params = {
            "expr": "((cancer) OR (neoplasm)) AND ((interventional) OR (treatment)) AND ((mutation) OR (variant))",
            "min_rnk": current_rank,
            "max_rnk": current_rank + page_size - 1,
            "fmt": "json",
            "fields": "NCTId"
        }
        response = requests.get(base_url, params=search_params)
        if response.status_code == 200:
            trials_data = response.json()
            if "FullStudiesResponse" in trials_data:
                studies = trials_data["FullStudiesResponse"].get("FullStudies", [])
                if not studies:
                    break  # No more studies found, exit the loop
                for study in studies:
                    nct_id = (
                        study.get("Study", {})
                             .get("ProtocolSection", {})
                             .get("IdentificationModule", {})
                             .get("NCTId")
                    )
                    if nct_id:
                        trials_set.add(nct_id)
                        trials_fetched += 1
                        if trials_fetched >= max_trials:
                            break
                current_rank += page_size
            else:
                logging.error("No trials found matching the criteria.")
                break
        else:
            logging.error("Failed to retrieve data. Status code: %s", response.status_code)
            break

    return list(trials_set)


def download_study_info(nct_id: str, delay: float = 1.0,
                        session: Union[requests.Session, None] = None) -> bool:
    """
    Download and update the XML information for a given clinical trial (NCT ID).
    
    If a local copy exists, compare selected fields with the online version.
    If differences are found (or the file doesn't exist), update or create the local file.
    
    Args:
        nct_id (str): The clinical trial NCT ID.
        delay (float): Delay between requests in seconds.
        session (requests.Session, optional): Requests session for connection pooling.
    
    Returns:
        bool: True if the file was downloaded or updated successfully; False otherwise.
    """
    local_file_path = os.path.join("..", "..", "data", "trials_xmls", f"{nct_id}.xml")
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    session = session or requests.Session()
    online_url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"

    try:
        response = session.get(online_url)
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching XML for trial %s: %s", nct_id, e)
        time.sleep(delay)
        return False

    if response.status_code != 200:
        logging.error("Error: received status code %s for trial %s", response.status_code, nct_id)
        time.sleep(delay)
        return False

    try:
        online_root = ET.fromstring(response.text)
    except ET.ParseError as e:
        logging.error("Error parsing online XML for trial %s: %s", nct_id, e)
        time.sleep(delay)
        return False

    # Check if a local version exists and compare key fields
    if os.path.exists(local_file_path):
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                local_xml_content = f.read()
            local_root = ET.fromstring(local_xml_content)
        except (ET.ParseError, IOError) as e:
            logging.error("Error reading/parsing local XML for trial %s: %s", nct_id, e)
            os.remove(local_file_path)
            local_root = None
    else:
        local_root = None

    fields_to_check = ["eligibility", "brief_title", "overall_status", "location"]
    needs_update = False

    if local_root is not None:
        for field in fields_to_check:
            local_elem = local_root.find(f".//{field}")
            online_elem = online_root.find(f".//{field}")
            if local_elem is not None and online_elem is not None:
                local_text = normalize_whitespace(ET.tostring(local_elem, encoding='unicode').strip())
                online_text = normalize_whitespace(ET.tostring(online_elem, encoding='unicode').strip())
                if local_text != online_text:
                    needs_update = True
                    break
            else:
                needs_update = True
                break

    if local_root is None or needs_update:
        try:
            with open(local_file_path, "w", encoding="utf-8") as f:
                f.write(ET.tostring(online_root, encoding='unicode'))
            if local_root is None:
                logging.info("Downloaded study information for %s", nct_id)
            else:
                logging.info("Updated study information for %s", nct_id)
        except IOError as e:
            logging.error("Error writing XML for trial %s: %s", nct_id, e)
            time.sleep(delay)
            return False
    else:
        logging.info("No changes in study information for %s", nct_id)

    time.sleep(delay)
    return True


def parallel_downloader(nct_ids: List[str], n_jobs: int = 10, delay: float = 1.0) -> List[bool]:
    """
    Download and update clinical trial XMLs in parallel using joblib.
    
    Args:
        nct_ids (List[str]): List of clinical trial NCT IDs.
        n_jobs (int): Number of parallel jobs.
        delay (float): Delay between requests.
    
    Returns:
        List[bool]: List indicating the success status of each download/update.
    """
    session = requests.Session()

    def download_wrapper(nct_id: str) -> bool:
        return download_study_info(nct_id, delay=delay, session=session)

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(download_wrapper)(nct_id) for nct_id in tqdm(nct_ids, desc="Downloading trials")
    )
    return results


class Downloader:
    """
    A class to manage the downloading and updating of clinical trial XML files.
    
    Attributes:
        id_list (List[str]): List of clinical trial NCT IDs.
        n_jobs (int): Number of parallel jobs.
        delay (float): Delay between requests.
    """
    
    def __init__(self, id_list: List[str], n_jobs: int = 10, delay: float = 1.0):
        self.id_list = id_list
        self.n_jobs = n_jobs
        self.delay = delay

    def download_and_update_trials(self) -> List[bool]:
        """
        Download and update XML files for all trials in the id_list.
        
        Returns:
            List[bool]: List of boolean statuses for each download/update.
        """
        start_time = time.time()
        results = parallel_downloader(self.id_list, n_jobs=self.n_jobs, delay=self.delay)
        elapsed_time = time.time() - start_time
        logging.info("Elapsed time: %.2f seconds", elapsed_time)
        return results


def main():
    """Main function to run the downloader."""
    id_file = 'nct_ids.txt'
    if not os.path.exists(id_file):
        logging.error("ID file '%s' does not exist.", id_file)
        sys.exit(1)
    
    with open(id_file, 'r', encoding="utf-8") as file:
        id_list = [line.strip() for line in file if line.strip()]

    if not id_list:
        logging.error("No NCT IDs found in the file.")
        sys.exit(1)
    
    n_jobs = 10
    downloader = Downloader(id_list, n_jobs=n_jobs, delay=1.0)
    downloader.download_and_update_trials()


if __name__ == "__main__":
    main()
