
import os
import joblib
from joblib import delayed
from tqdm.auto import tqdm
import requests
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import glob
import json

import torch
import medspacy
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import filter_spans
from spacy.tokens import Doc, Token
from spacy.matcher import Matcher
from srsly import read_json
import re
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings

from biomedner_engine import BioMedNER
# # Filepaths
INPUT_FILEPATH = '/home/mabdallah/TrialMatchAI/data/preprocessed_data'
OUTPUT_FILEPATH_CT = '/home/mabdallah/TrialMatchAI/data/ner_trial/'
# # OUTPUT_FILEPATH_PAT = "../data/ner_patients_clinical_notes/"


biomedner = BioMedNER(
    max_word_len=50,
    seed=2019,
    gnormplus_home="resources/GNormPlusJava/",
    gnormplus_host="localhost",
    gnormplus_port=18895,
    gene_norm_port=18888,
    disease_norm_port=18892,
    biomedner_home=".",
    biomedner_host="localhost",
    biomedner_port=18894,
    maccrobat_host="localhost",
    maccrobat_port=18783,
    time_format='[%d/%b/%Y %H:%M:%S.%f]',
    use_neural_normalizer=True,
    keep_files=True,
    no_cuda=False,
)

# Memory caching for function calls
memory = joblib.Memory(".")

def ParallelExecutor(use_bar="tqdm", **joblib_args):
    """
    Utility function for tqdm progress bar in joblib.Parallel.

    This function is a utility for using tqdm progress bar with joblib.Parallel. It returns a function that can be used as a wrapper
    for the operation iterator in joblib.Parallel. The function takes a 'bar' argument which specifies the type of progress bar to use.
    The available options are 'tqdm', 'False', and 'None'. The function also accepts additional arguments that are passed to tqdm.

    Parameters:
        use_bar (str): The type of progress bar to use. Default is "tqdm".
        **tq_args: Additional arguments to be passed to tqdm.

    Returns:
        function: The wrapper function that can be used with joblib.Parallel.

    Example:
        executor = ParallelExecutor(use_bar="tqdm", ncols=80)
        results = executor(op_iter)
    """
    all_bar_funcs = {
        "tqdm": lambda args: lambda x: tqdm(x, **args),
        "False": lambda args: lambda x: x,
        "None": lambda args: lambda x: x,
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

def merge_similar_consecutive_entities(entities):
    if not entities:
        return []

    # Define groups of interest directly as a set for faster membership testing
    groups_of_interest = {"Lab_value", "Diagnostic_procedure", "Therapeutic_procedure", "Detailed_description"}
    combined_entities = []
    current_entity = entities[0]

    # Ensure the current entity meets basic structure requirements
    necessary_keys = {'text', 'entity_group', 'start', 'end'}
    if not all(key in current_entity for key in necessary_keys):
        raise ValueError("Each entity must have 'text', 'entity_group', 'start', and 'end' keys.")

    for next_entity in entities[1:]:
        if not all(key in next_entity for key in necessary_keys):
            raise ValueError("Each entity must have 'text', 'entity_group', 'start', and 'end' keys.")

        # Check if both entities are in the groups of interest and are consecutive or nearly so (gap up to 3 characters)
        if (current_entity['entity_group'] == next_entity['entity_group'] and
            current_entity['entity_group'] in groups_of_interest and
            0 <= next_entity['start'] - current_entity['end'] - 1 <= 3):

            # Merge entities by updating the 'text' and 'end' of the current entity
            current_entity['text'] += ' ' + next_entity['text']
            current_entity['end'] = next_entity['end']
        else:
            # If not mergeable, add current to combined and move to next
            combined_entities.append(current_entity)
            current_entity = next_entity

    # Append the last processed entity
    combined_entities.append(current_entity)
    return combined_entities


class EntityRecognizer:
    def __init__(self, id_list, n_jobs, data_source="clinical trials"):
        self.id_list = id_list
        self.n_jobs = n_jobs
        self.data_source = data_source
        
    def data_loader(self, id_list):
        to_concat = []
        for idx in id_list:
            if self.data_source == "clinical trials":
                file_path = os.path.join(INPUT_FILEPATH, "clinical_trials", f"{idx}_preprocessed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    to_concat.append(df)
            elif self.data_source=="patient":
                df = pd.read_csv(INPUT_FILEPATH + "patient/" + "%s_preprocessed.csv"%idx)
                to_concat.append(df)
            else:
                warnings.warn("Unexpected data source encountered. Please choose between 'clinical trials' or 'patient'", UserWarning)
        return to_concat

    def recognize_entities(self, df):
        df = df.dropna()
        df = df.rename(columns={"Unnamed: 0": "index"})
        df = df.loc[df['index'] != 0]
        criteria_list = []
        trans_table = str.maketrans({
            '"': '',
            '{': '(',
            '}': ')'})
        for _,row in df.iterrows():
            sentence = row["sentence"].translate(trans_table)
            entities = biomedner.annotate_text(sentence)
            entities = merge_similar_consecutive_entities(entities)
            if len(entities) > 0:
                for e in entities:
                    e["entity"] = e.pop("text")
                    e["class"] = e.pop("entity_group")
                    for key in ["start", "end", "score"]:
                        e.pop(key, None) 
            criteria_list.append({"criterion": sentence, "entities": entities, "type": row["criteria"]})
        trial_json = {"nct_id": df["id"].iloc[0], "criteria": criteria_list}
        return trial_json
    
    def save_json_output(self, dict, output_filepath):
        with open(output_filepath, 'w') as f:
            json.dump(dict, f,  indent=4)

    def __call__(self):
        all_df = self.data_loader(self.id_list)

        def process_dataframe(df):
            output_filepath = OUTPUT_FILEPATH_CT + df["id"].iloc[0] + ".json"
            if not os.path.exists(output_filepath):
                result_json = self.recognize_entities(df)
                if self.data_source == "clinical trials":
                    self.save_json_output(result_json, output_filepath)
                
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        
        parallel_runner(delayed(process_dataframe)(df) for df in all_df)
        
        return 

if __name__ == "__main__":
    # Load the list of NCT IDs
    folder_path = '/home/mabdallah/TrialMatchAI/data/trials_xmls' 
    file_names = []
    # List all files in the folder
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_name, file_extension = os.path.splitext(file)
            file_names.append(file_name)
    nct_ids = file_names
    reco = EntityRecognizer(n_jobs=5, id_list=nct_ids, data_source="clinical trials")
    entities = reco()