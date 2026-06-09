import json
import multiprocessing
import os
import re
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from biomedner_engine import BioMedNER

# Filepaths
BASE_INPUT_FILEPATH = os.path.join(
    os.path.dirname(__file__), "../../data/preprocessed_data"
)
BASE_OUTPUT_FILEPATH_CT = os.path.join(
    os.path.dirname(__file__), "../../data/parsed_trec"
)
BASE_TMP_SUPERDIR = os.path.join(os.path.dirname(__file__), "tmp")
DICT_PATH = Path("source/Parser/resources/normalization/dictionary")
dict_paths = {
    "gene": DICT_PATH / "dict_Gene.txt",
    "disease": DICT_PATH / "dict_Disease_20210630.txt",
    "cell type": DICT_PATH / "dict_CellType_20210810.txt",
    "drug": DICT_PATH / "dict_ChemicalCompound_20210630.txt",
    "procedure": DICT_PATH / "dict_Procedures.txt",
    "sign symptom": DICT_PATH / "dict_SignSymptom.txt",
}


def load_shared_dictionaries():
    return {
        key: BioMedNER.load_dictionary_file(path) for key, path in dict_paths.items()
    }


# Global variable for BioMedNER instance (initialized in each process)
biomedner = None
data_source = "clinical trials"  # Adjust this based on your data source


def process_dataframe(df, text_column):
    new_data = []
    for _, row in df.iterrows():
        row_copy = row.to_dict()
        text = row_copy.pop(text_column)

        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)  # Convert to string if not already

        word_count = len(text.split())
        sentence_end_count = text.count(".") + text.count("?") + text.count("!")

        if word_count > 512 and sentence_end_count > 1:
            sentences = re.split(r"(?<=[.!?]) +", text)
            for sentence in sentences:
                new_row = row_copy.copy()
                new_row[text_column] = sentence
                new_data.append(new_row)
        else:
            row_copy[text_column] = text
            new_data.append(row_copy)

    new_df = pd.DataFrame(new_data)
    return new_df


def process_file(idx):
    global biomedner
    if biomedner is None:
        raise RuntimeError(
            "BioMedNER instance is not initialized. Please ensure 'biomedner' is set before calling process_file."
        )
    output_filepath = os.path.join(BASE_OUTPUT_FILEPATH_CT, f"{idx}.json")
    if os.path.exists(output_filepath):
        print(f"File {idx} already exists. Skipping...")
        return

    # Load the dataframe
    file_path = None
    try:
        if data_source == "clinical trials":
            file_path = os.path.join(
                BASE_INPUT_FILEPATH, "clintra", f"{idx}_preprocessed.tsv"
            )
            if os.path.exists(file_path):
                print(f"Processing {idx}")
                df = pd.read_csv(file_path, delimiter="\t")
            else:
                print(f"File {file_path} does not exist. Skipping.")
                return
        elif data_source == "patient":
            file_path = os.path.join(
                BASE_INPUT_FILEPATH, "patient", f"{idx}_preprocessed.csv"
            )
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                print(f"File {file_path} does not exist. Skipping.")
                return
        else:
            warnings.warn(
                "Unexpected data source encountered. Please choose between 'clinical trials' or 'patient'",
                UserWarning,
            )
            return
    except pd.errors.EmptyDataError:
        print(
            f"EmptyDataError: No columns to parse from file {file_path}. Skipping this file."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing file {file_path}: {e}")
        return

    if df.empty:
        print(f"Dataframe {idx} is empty. Skipping.")
        return

    print(f"Processing {idx}")
    df = process_dataframe(df, "sentence")
    sentences = df["sentence"].tolist()
    if not sentences:
        print(f"No sentences to process for {idx}. Skipping.")
        return

    # Use the biomedner instance
    annotated_sentences = biomedner.annotate_texts_in_parallel(sentences, max_workers=5)
    df["entities"] = [entities for entities in annotated_sentences]
    result_json = {
        "nct_id": idx,
        "criteria": [
            {
                "criterion": row["sentence"],
                "entities": row["entities"],
                "type": row["criteria"],
            }
            for _, row in df.iterrows()
        ],
    }
    # Save the result
    with open(output_filepath, "w") as f:
        json.dump(result_json, f, indent=4)
    print(f"Saved {output_filepath}")


def process_files(device_id, ids_to_process, biomedner_params, shared_dicts):
    import os

    global biomedner

    # Set the environment variable for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # Initialize BioMedNER instance
    biomedner = BioMedNER(
        **biomedner_params,
        biomedner_home=tempfile.TemporaryDirectory(dir=BASE_TMP_SUPERDIR).name,
        preloaded_dicts=shared_dicts,
    )

    # Process each file in the chunk
    for idx in ids_to_process:
        process_file(idx)


if __name__ == "__main__":
    import multiprocessing

    # Load dictionaries once
    shared_dicts = load_shared_dictionaries()

    # Define BioMedNER parameters
    biomedner_params = {
        "max_word_len": 50,
        "seed": 2019,
        "gene_norm_port": 18888,
        "disease_norm_port": 18892,
        "biomedner_host": "localhost",
        "biomedner_port": 18894,
        "gner_host": "localhost",
        "gner_port": 18783,
        "time_format": "[%d/%b/%Y %H:%M:%S.%f]",
        "use_neural_normalizer": True,
        "no_cuda": False,
    }

    # Load your NCT IDs
    df_trec21 = pd.read_csv("../../data/trec/Unique_NCT_IDs_from_2021_File.csv")
    df_trec22 = pd.read_csv("../../data/trec/Unique_NCT_IDs_from_2022_File.csv")
    nct_ids21 = df_trec21["Unique NCT IDs"].unique().tolist()
    nct_ids22 = df_trec22["Unique NCT IDs"].unique().tolist()
    unique_ids = set(nct_ids21 + nct_ids22)
    # Prepare the list of unprocessed ids
    unprocessed_ids = [
        idx
        for idx in unique_ids
        if not os.path.exists(os.path.join(BASE_OUTPUT_FILEPATH_CT, f"{idx}.json"))
    ]

    # Define your device IDs
    device_ids = ["0"]  # List of GPU IDs as strings

    # Define the number of processes you want
    num_processes = 8  # For example, 12 processes

    # Create a list of device IDs cycling through the available GPUs
    process_device_ids = [device_ids[i % len(device_ids)] for i in range(num_processes)]

    # Divide unprocessed_ids among processes
    chunks = [[] for _ in range(num_processes)]
    for i, idx in enumerate(unprocessed_ids):
        chunks[i % num_processes].append(idx)

    processes = []
    for i in range(num_processes):
        device_id = process_device_ids[i]
        ids_chunk = chunks[i]
        if ids_chunk:  # Only start a process if there's work to do
            p = multiprocessing.Process(
                target=process_files,
                args=(device_id, ids_chunk, biomedner_params, shared_dicts),
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
