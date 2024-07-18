import os
import pandas as pd
import json
import unicodedata
import re
import warnings
from biomedner_engine import BioMedNER
import gc
import warnings
import string

warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.")

# Filepaths
INPUT_FILEPATH = '../../data/preprocessed_data'
OUTPUT_FILEPATH_CT = '../../data/parsed_trials/'

biomedner = BioMedNER(
    max_word_len=50,
    seed=2019,
    gene_norm_port=18888,
    disease_norm_port=18892,
    biomedner_home=".",
    biomedner_host="localhost",
    biomedner_port=18894,
    gner_host="localhost",
    gner_port=18783,
    time_format='[%d/%b/%Y %H:%M:%S.%f]',
    use_neural_normalizer=True,
    no_cuda=False,
)
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
    def __init__(self, id_list, data_source="clinical trials"):
        self.id_list = id_list
        self.data_source = data_source

    def data_loader(self, id_list):
        for idx in id_list:
            if self.data_source == "clinical trials":
                file_path = os.path.join(INPUT_FILEPATH, "clinical_trials", f"{idx}_preprocessed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    yield df
            elif self.data_source == "patient":
                file_path = os.path.join(INPUT_FILEPATH, "patient", f"{idx}_preprocessed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    yield df
            else:
                warnings.warn("Unexpected data source encountered. Please choose between 'clinical trials' or 'patient'", UserWarning)

    def clean_entity(self, entity):
        # Remove unnecessary white space and punctuation
        entity = entity.strip()  # Remove leading/trailing white space
        entity = re.sub(r'\s+', ' ', entity)  # Replace multiple spaces with a single space
        entity = entity.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        entity = re.sub(r'[\(\)\[\]\{\}]', '', entity)  # Remove brackets
        return entity

    def recognize_entities(self, df):
        print(f"Now Processing {df['id'].iloc[0]}")
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
        annotated_sentences = biomedner.annotate_texts_in_parallel(sentences, max_workers=5)

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
        trial_json = {"nct_id": df["id"].iloc[0], "criteria": criteria_list}
        del df
        gc.collect()
        return trial_json

    def save_json_output(self, dict, output_filepath):
        with open(output_filepath, 'w') as f:
            json.dump(dict, f, indent=4)

    def __call__(self):
        for df in self.data_loader(self.id_list):
            output_filepath = OUTPUT_FILEPATH_CT + df["id"].iloc[0] + ".json"
            if os.path.exists(output_filepath):
                print(f"File {df['id'].iloc[0]} already exists. Skipping...")
                continue
            else:
                result_json = self.recognize_entities(df)
                if self.data_source == "clinical trials":
                    self.save_json_output(result_json, output_filepath)
            del df
            gc.collect()

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
    reco = EntityRecognizer(id_list=nct_ids, data_source="clinical trials")
    entities = reco()