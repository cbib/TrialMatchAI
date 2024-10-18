import os
import pandas as pd
import json
import unicodedata
import re
import warnings
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import tempfile
from biomedner_engine import BioMedNER
import gc

warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")
warnings.filterwarnings("ignore", message="resume_download is deprecated and will be removed in version 1.0.0.")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.")

# Filepaths
BASE_INPUT_FILEPATH = os.path.join(os.path.dirname(__file__), '../../data/preprocessed_data')
BASE_OUTPUT_FILEPATH_CT = os.path.join(os.path.dirname(__file__), '../../data/parsed_trials/')
BASE_TMP_SUPERDIR = os.path.join(os.path.dirname(__file__), 'tmp')

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
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!) ')
    return len(sentence_endings.findall(text))

def split_text_into_sentences(text):
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
    def __init__(self, id_list, biomedner_params, data_source="clinical trials"):
        self.id_list = id_list
        self.biomedner_params = biomedner_params
        self.data_source = data_source

    def data_loader(self, id_list):
        for idx in id_list:
            if self.data_source == "clinical trials":
                file_path = os.path.join(BASE_INPUT_FILEPATH, "clinical_trials", f"{idx}_preprocessed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    yield idx, df
            elif self.data_source == "patient":
                file_path = os.path.join(BASE_INPUT_FILEPATH, "patient", f"{idx}_preprocessed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    yield idx, df
            else:
                warnings.warn("Unexpected data source encountered. Please choose between 'clinical trials' or 'patient'", UserWarning)

    def clean_entity(self, entity):
        entity = entity.strip()
        entity = re.sub(r'\s+', ' ', entity)
        entity = entity.translate(str.maketrans('', '', string.punctuation))
        entity = re.sub(r'[\(\)\[\]\{\}]', '', entity)
        return entity

    def recognize_entities(self, idx, df, biomedner):
        if df.empty:
            print(f"Dataframe {idx} is empty. Skipping.")
            return None

        print(f"Now Processing {idx}")
        df = df.dropna()
        df = process_dataframe(df, "sentence")
        print(f"Processed dataframe for {idx}: {df.shape[0]} sentences")
        criteria_list = []
        trans_table = str.maketrans(
            {
            '"': '',
            '{': '(',
            '}': ')'})
        df["sentence"] = df["sentence"].apply(lambda x: x.translate(trans_table))
        df["sentence"] = df["sentence"].apply(replace_unicode_symbols)

        sentences = df["sentence"].tolist()
        if not sentences:
            print(f"No sentences to process for {idx}. Skipping.")
            return None

        print(f"Annotating {len(sentences)} sentences for {idx}")
        annotated_sentences = biomedner.annotate_texts_in_parallel(sentences, max_workers=15)
        if not annotated_sentences:
            print(f"No annotated sentences returned for {idx}. Skipping.")
            return None

        df["entities"] = [entities for entities in annotated_sentences]
        
        for _, row in df.iterrows():
            sentence = row["sentence"].translate(trans_table)
            entities = row["entities"]
            if len(entities) > 0:
                for e in entities:
                    e["entity"] = self.clean_entity(e.pop("text"))
                    e["class"] = e.pop("entity_group")
                    for key in ["start", "end", "score"]:
                        e.pop(key, None)
            criteria_list.append({"criterion": sentence, "entities": entities, "type": row["criteria"]})
        trial_json = {"nct_id": idx, "criteria": criteria_list}
        del df
        return trial_json

    def save_json_output(self, dict, output_filepath):
        with open(output_filepath, 'w') as f:
            json.dump(dict, f, indent=4)

    def process_single_file(self, idx, df, biomedner_params):
        # Check if the output file already exists
        output_filepath = os.path.join(BASE_OUTPUT_FILEPATH_CT, f"{idx}.json")
        if os.path.exists(output_filepath):
            print(f"File {idx} already exists. Skipping...")
            return

        # Create the temporary super-directory if it doesn't exist
        os.makedirs(BASE_TMP_SUPERDIR, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=BASE_TMP_SUPERDIR) as temp_dir:
            biomedner_home = temp_dir  # Set the biomedner_home to the temporary directory

            print(f"Processing {idx} with temporary directory {biomedner_home}")
            biomedner = BioMedNER(**biomedner_params, biomedner_home=biomedner_home)  # Initialize with the temporary biomedner_home
            
            result_json = self.recognize_entities(idx, df, biomedner)
            if result_json:
                self.save_json_output(result_json, output_filepath)
                print(f"Saved {output_filepath}")
            else:
                print(f"No valid results for {idx}. Skipping.")
                
            del biomedner
            gc.collect()

    def __call__(self):
        # Filter out already processed files
        unprocessed_ids = []
        skipped_files = []
        for idx in self.id_list:
            output_filepath = os.path.join(BASE_OUTPUT_FILEPATH_CT, f"{idx}.json")
            if os.path.exists(output_filepath):
                skipped_files.append(idx)
            else:
                unprocessed_ids.append(idx)
        
        # Print skipped files
        print(f"Skipped files: {skipped_files}")

        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [
                executor.submit(self.process_single_file, idx, df, self.biomedner_params)
                for idx, df in self.data_loader(unprocessed_ids)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    # Load the list of NCT IDs
    folder_path = '../../data/trials_xmls' 
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_name, file_extension = os.path.splitext(file)
            file_names.append(file_name)
    nct_ids = file_names

    # Define BioMedNER parameters
    biomedner_params = {
        'max_word_len': 50,
        'seed': 2019,
        'gene_norm_port': 18888,
        'disease_norm_port': 18892,
        'biomedner_host': "localhost",
        'biomedner_port': 18894,
        'gner_host': "localhost",
        'gner_port': 18783,
        'time_format': '[%d/%b/%Y %H:%M:%S.%f]',
        'use_neural_normalizer': True,
        'no_cuda': False,
    }

    reco = EntityRecognizer(id_list=nct_ids, biomedner_params=biomedner_params, data_source="clinical trials")
    reco()
