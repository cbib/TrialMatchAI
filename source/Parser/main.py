from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from Matcher.config.config_loader import load_config  # noqa: E402
from Parser.biomedner_engine import BioMedNER  # noqa: E402


BASE_INPUT_FILEPATH = Path(__file__).resolve().parents[2] / "data/preprocessed_data"
BASE_OUTPUT_FILEPATH_CT = Path(__file__).resolve().parents[2] / "data/parsed_trec"

entity_annotator: BioMedNER | None = None
data_source = "clinical trials"


def process_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    new_data = []
    for _, row in df.iterrows():
        row_copy = row.to_dict()
        text = row_copy.pop(text_column)
        if not isinstance(text, str):
            text = str(text)

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

    return pd.DataFrame(new_data)


def process_file(idx: str) -> None:
    global entity_annotator
    if entity_annotator is None:
        raise RuntimeError("Entity annotator is not initialized.")

    output_filepath = BASE_OUTPUT_FILEPATH_CT / f"{idx}.json"
    if output_filepath.exists():
        print(f"File {idx} already exists. Skipping...")
        return

    df = _load_dataframe(idx)
    if df is None or df.empty:
        print(f"Dataframe {idx} is empty. Skipping.")
        return

    print(f"Processing {idx}")
    df = process_dataframe(df, "sentence")
    sentences = df["sentence"].tolist()
    if not sentences:
        print(f"No sentences to process for {idx}. Skipping.")
        return

    annotated_sentences = entity_annotator.annotate_texts_in_parallel(
        sentences,
        max_workers=5,
    )
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
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_filepath.write_text(json.dumps(result_json, indent=4))
    print(f"Saved {output_filepath}")


def process_files(
    device_id: str,
    ids_to_process: list[str],
    config_path: str | None,
) -> None:
    global entity_annotator
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    config = load_config(config_path) if config_path else None
    entity_annotator = BioMedNER(config=config)
    for idx in ids_to_process:
        process_file(idx)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate preprocessed trial criteria with TrialMatchAI entities."
    )
    parser.add_argument("--config", default=None, help="TrialMatchAI config path")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--device-id", action="append", default=["0"])
    parser.add_argument(
        "--ids-file",
        default=None,
        help="Optional file containing one NCT ID per line. Defaults to TREC ID files.",
    )
    args = parser.parse_args()

    unique_ids = _load_ids(args.ids_file)
    unprocessed_ids = [
        idx for idx in unique_ids if not (BASE_OUTPUT_FILEPATH_CT / f"{idx}.json").exists()
    ]
    if not unprocessed_ids:
        print("No unprocessed IDs found.")
        return 0

    num_processes = max(1, args.processes)
    process_device_ids = [
        args.device_id[i % len(args.device_id)] for i in range(num_processes)
    ]
    chunks = [[] for _ in range(num_processes)]
    for i, idx in enumerate(unprocessed_ids):
        chunks[i % num_processes].append(idx)

    processes = []
    for i, ids_chunk in enumerate(chunks):
        if not ids_chunk:
            continue
        process = multiprocessing.Process(
            target=process_files,
            args=(process_device_ids[i], ids_chunk, args.config),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode:
            return process.exitcode
    return 0


def _load_dataframe(idx: str) -> pd.DataFrame | None:
    if data_source == "clinical trials":
        file_path = BASE_INPUT_FILEPATH / "clintra" / f"{idx}_preprocessed.tsv"
        delimiter = "\t"
    elif data_source == "patient":
        file_path = BASE_INPUT_FILEPATH / "patient" / f"{idx}_preprocessed.csv"
        delimiter = ","
    else:
        raise ValueError("data_source must be 'clinical trials' or 'patient'.")

    if not file_path.exists():
        print(f"File {file_path} does not exist. Skipping.")
        return None
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except pd.errors.EmptyDataError:
        print(f"No columns to parse from file {file_path}. Skipping.")
        return None


def _load_ids(ids_file: str | None) -> list[str]:
    if ids_file:
        return [
            line.strip()
            for line in Path(ids_file).read_text().splitlines()
            if line.strip()
        ]
    trec_root = Path(__file__).resolve().parents[2] / "data/trec"
    df_trec21 = pd.read_csv(trec_root / "Unique_NCT_IDs_from_2021_File.csv")
    df_trec22 = pd.read_csv(trec_root / "Unique_NCT_IDs_from_2022_File.csv")
    nct_ids21 = df_trec21["Unique NCT IDs"].unique().tolist()
    nct_ids22 = df_trec22["Unique NCT IDs"].unique().tolist()
    return sorted(set(nct_ids21 + nct_ids22))


if __name__ == "__main__":
    raise SystemExit(main())
