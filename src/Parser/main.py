import os
import joblib
from joblib import delayed
from tqdm.auto import tqdm
import pandas as pd
import json
import unicodedata
import re
import warnings

from biomedner_engine import BioMedNER, annotate_texts_in_parallel

# Filepaths
INPUT_FILEPATH = '/home/mabdallah/TrialMatchAI/data/preprocessed_data'
OUTPUT_FILEPATH_CT = '/home/mabdallah/TrialMatchAI/data/ner_trial/'

biomedner = BioMedNER(
    max_word_len=50,
    seed=2019,
    gene_norm_port=18888,
    disease_norm_port=18892,
    biomedner_home=".",
    biomedner_host="localhost",
    biomedner_port=18894,
    maccrobat_host="localhost",
    maccrobat_port=18783,
    time_format='[%d/%b/%Y %H:%M:%S.%f]',
    use_neural_normalizer=True,
    no_cuda=False,
)

# Memory caching for function calls
import joblib
from tqdm import tqdm

memory = joblib.Memory(".")

def ParallelExecutor(use_bar="tqdm", **joblib_args):
    """
    Utility function for tqdm progress bar in joblib.Parallel.
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
            # Pass n_jobs from joblib_args and set the backend to 'threading'
            return joblib.Parallel(n_jobs=joblib_args.get("n_jobs", 10), backend='threading')(bar_func(op_iter))

        return tmp

    return aprun


def merge_similar_consecutive_entities(entities):
    if not entities:
        return []

    groups_of_interest = {"Lab_value", "Diagnostic_procedure", "Therapeutic_procedure", "Detailed_description"}
    combined_entities = []
    current_entity = entities[0]

    necessary_keys = {'text', 'entity_group', 'start', 'end'}
    if not all(key in current_entity for key in necessary_keys):
        raise ValueError("Each entity must have 'text', 'entity_group', 'start', and 'end' keys.")

    for next_entity in entities[1:]:
        if not all(key in next_entity for key in necessary_keys):
            raise ValueError("Each entity must have 'text', 'entity_group', 'start', and 'end' keys.")

        if (current_entity['entity_group'] == next_entity['entity_group'] and
            current_entity['entity_group'] in groups_of_interest and
            0 <= next_entity['start'] - current_entity['end'] - 1 <= 3):

            current_entity['text'] += ' ' + next_entity['text']
            current_entity['end'] = next_entity['end']
        else:
            combined_entities.append(current_entity)
            current_entity = next_entity

    combined_entities.append(current_entity)
    return combined_entities

def replace_unicode_symbols(text):
    def unicode_to_readable(match):
        char = match.group(0)
        try:
            name = unicodedata.name(char).lower() + ' '
            return name
        except ValueError:
            return char

    return re.sub(r'[\u0080-\uFFFF]', unicode_to_readable, text)

def count_sentence_ending_fullstops(text):
    # Regular expression to match sentence-ending full stops
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!) ')
    return len(sentence_endings.findall(text))

def split_text_into_sentences(text):
    # Regular expression to split text on sentence-ending punctuation
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences

def process_dataframe(df, text_column):
    new_data = []
    for _, row in df.iterrows():
        row_copy = row.to_dict()
        text = row_copy.pop(text_column)
        word_count = len(text.split())
        sentence_end_count = count_sentence_ending_fullstops(text)
        if word_count > 256 and sentence_end_count > 1:
            sentences = split_text_into_sentences(text)
            for sentence in sentences:
                new_row = row_copy.copy()
                new_row[text_column] = sentence
                new_data.append(new_row)
        else:
            row_copy[text_column] = text
            new_data.append(row_copy)
    
    new_df = pd.DataFrame(new_data)
    return new_df

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
        print(f"Processing {df['id'].iloc[0]}")
        df = df.dropna()
        df = df.rename(columns={"Unnamed: 0": "index"})
        df = df.loc[df['index'] != 0]
        df = process_dataframe(df, "sentence")
        criteria_list = []
        trans_table = str.maketrans(
            {
            '"': '',
            '{': '(',
            '}': ')'})
        df["sentence"] = df["sentence"].apply(lambda x: x.translate(trans_table))
        df["sentence"] = df["sentence"].apply(replace_unicode_symbols)

        # Convert sentences to a list
        sentences = df["sentence"].tolist()

        # Annotate sentences in parallel
        annotated_sentences = annotate_texts_in_parallel(biomedner, sentences)

        # Merge similar consecutive entities
        merged_entities = [merge_similar_consecutive_entities(entities) for entities in annotated_sentences]
        
        # Assign the merged entities back to the dataframe
        df["entities"] = merged_entities

        for _, row in df.iterrows():
            sentence = row["sentence"].translate(trans_table)
            entities = row["entities"]
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
            json.dump(dict, f, indent=4)

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
    reco = EntityRecognizer(n_jobs=1, id_list=nct_ids, data_source="clinical trials")
    entities = reco()
