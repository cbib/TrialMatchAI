import requests
import sys
import xml.etree.ElementTree as ET
import os
import time
import joblib 
from tqdm.auto import tqdm
import numpy as np

# Open the log file
log_file = open('../logs/download.log', 'w')
# Redirect standard output to the log file
sys.stdout = log_file

def normalize_whitespace(s):
    return ' '.join(s.split())

def get_cancer_trials_list(max_trials=15000):
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
                studies = trials_data["FullStudiesResponse"]["FullStudies"]
                if not studies:
                    break  # No more studies found, exit the loop
                for study in studies:
                    trials_set.add(study["Study"]["ProtocolSection"]["IdentificationModule"]["NCTId"])
                    trials_fetched += 1
                    if trials_fetched == max_trials:
                        break
                current_rank += page_size
            else:
                print("No trials found matching the criteria.")
                break
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
            break

    return list(trials_set)  # Convert set to list for output
    

def download_study_info(nct_id):
    local_file_path = f"../data/trials_xmls/{nct_id}.xml"

    if os.path.exists(local_file_path):
        # Read the content of the existing local XML file
        with open(local_file_path, "r") as f:
            local_xml_content = f.read()
        try:
            local_root = ET.fromstring(local_xml_content)
        except ET.ParseError as e:
            print(f"Error parsing XML for trial {nct_id}: {e}")
            os.remove(local_file_path)

        
        # Download the online version of the XML
        url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching XML for trial {nct_id}: {e}")

        if response.status_code == 200:
            online_xml_content = response.text
            # Parse the XML content
            try:
                online_root = ET.fromstring(online_xml_content)
            except ET.ParseError as e:
                print(f"Error parsing online XML for trial {nct_id}: {e}")

        else:
            print(f"Error: received status code {response.status_code} when fetching XML for trial {nct_id}")

        to_check = ["eligibility", "brief_title", "overall_status", "location"]
                
        local_version = []
        online_version = []
        
        for s in to_check:
            local_elem = local_root.find(".//%s" % s)
            online_elem = online_root.find(".//%s" % s)
            
            # Check if the element exists in both versions
            if local_elem is not None and online_elem is not None:
                local_version.append(local_elem)
                online_version.append(online_elem)
            else:
                continue
        
        is_updated = any([normalize_whitespace(ET.tostring(a, encoding='unicode').strip()) !=
                        normalize_whitespace(ET.tostring(b, encoding='unicode').strip())
                        for a, b in zip(local_version, online_version)])

        if is_updated:
            # Update the local XML with the online version
            with open(local_file_path, "w") as f:
                f.write(ET.tostring(online_root, encoding='unicode'))
            print(f"Updated eligibility criteria for {nct_id}")
        else:
            print(f"No changes in eligibility criteria for {nct_id}.")
    else:
        # If the local file doesn't exist, download the online version
        url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching XML for trial {nct_id}: {e}")

        if response.status_code == 200:
            try:
                root = ET.fromstring(response.text)
                with open(local_file_path, "w") as f:
                    f.write(ET.tostring(root, encoding='unicode'))
                print(f"Study information downloaded for {nct_id}")
            except ET.ParseError as e:
                print(f"Error parsing online XML for trial {nct_id}: {e}")
        else:
            print(f"Error: received status code {response.status_code} when fetching XML for trial {nct_id}")
    return []

    

memory = joblib.Memory(".")
def ParallelExecutor(use_bar="tqdm", **joblib_args):
    """Utility for tqdm progress bar in joblib.Parallel"""
    all_bar_funcs = {
        "tqdm": lambda args: lambda x: tqdm(x, **args),
        "False": lambda args: iter,
        "None": lambda args: iter,
    }
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            
            # Pass n_jobs from joblib_args
            return joblib.Parallel(n_jobs=joblib_args.get("n_jobs", 10))(bar_func(op_iter))

        return tmp
    return aprun

def parallel_downloader(
    n_jobs,
    nct_ids,
):
    parallel_runner = ParallelExecutor(n_jobs=n_jobs)(total=len(nct_ids))
    X = parallel_runner(
        joblib.delayed(download_study_info)(
        nct_id, 
        )
        for nct_id in nct_ids
    )     
    updated_cts = np.vstack(X).flatten()
    return updated_cts 


class Downloader:
    def __init__(self, id_list, n_jobs):
        self.id_list = id_list
        self.n_jobs = n_jobs

    def download_and_update_trials(self):
        start_time = time.time()
        updated_cts = parallel_downloader(self.n_jobs, self.id_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return updated_cts
    
    
    