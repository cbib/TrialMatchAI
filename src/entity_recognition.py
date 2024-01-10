import os
import joblib
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
from transformers import AutoTokenizer, pipeline
import warnings

INPUT_FILEPATH = "../data/preprocessed_data/"
OUTPUT_FILEPATH_CT = "../data/ner_clinical_trials/"
OUTPUT_FILEPATH_PAT = "../data/ner_patients_clinical_notes/"
AUXILIARY_ENTITIES_LIST = ["Disease_disorder", "Sign_symptom", "Biological_structure", "Date", "Duration", "Time", "Frequency", "Severity", "Lab_value", "Dosage",
                           "Diagnostic_procedure", "Therapeutic_procedure", "Medication", "Clinical_event", "Outcome", "History", "Subject,"
                           "Family_history", "Detailed_description", "Area"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aux_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
aux_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer= aux_tokenizer, aggregation_strategy="first", device=device)
mutations_tokenizer = AutoTokenizer.from_pretrained("Brizape/tmvar-PubMedBert-finetuned-24-02")
mutations_pipeline = pipeline("ner", model="Brizape/tmvar-PubMedBert-finetuned-24-02", tokenizer=mutations_tokenizer, aggregation_strategy="first",  device=device)

def query_plain(text, url="http://localhost:8888/plain"):
    return json.loads(requests.post(url, json={'text': text}).content.decode('utf-8'))

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

def get_dictionaries_with_values(list_of_dicts, key, values):
    """
    Filter a list of dictionaries based on the presence of specific values in a specified key.

    This function takes a list of dictionaries and filters them based on the presence of specific values in a specified key.
    The function checks each dictionary in the input list and includes only those dictionaries where any of the given values
    are present in the specified key. The filtering is performed using list comprehensions.

    Parameters:
        list_of_dicts (list): A list of dictionaries to be filtered.
        key (str): The key in the dictionaries where the filtering is applied.
        values (list): A list of values. The function will filter dictionaries where any of these values are present in the specified key.

    Returns:
        list: A list of dictionaries that meet the filtering criteria.

    Example:
        list_of_dicts = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
            {"name": "David", "age": 30},
        ]

        get_dictionaries_with_values(list_of_dicts, "age", [30, 35])
        # Output: [
        #   {"name": "Alice", "age": 30},
        #   {"name": "Charlie", "age": 35},
        #   {"name": "David", "age": 30}
        # ]
    """
    return [d for d in list_of_dicts if any(val in d.get(key, []) for val in values)]


def add_custom_entity(doc, entity):
    entity["text"] = re.sub(r'([,.-])\s+', r'\1', entity["text"]) 
    # print(entity["text"])
    entity_text = entity["text"].lower()
    start_char = entity["start"] 
    end_char = entity["end"] 
    # Find the token indices corresponding to the character span
    start_indices = [i for i, token in enumerate(doc) if (start_char <= token.idx <= end_char) or (entity_text in token.text and token.idx <= start_char)]
    if start_indices:
    # You can choose the first matching window or handle multiple matches
        start_index = start_indices[0]
        start_token = doc[start_index]
        end_index = min(start_index + len(entity_text.split()) - 1, len(doc) - 1)
        end_token = doc[end_index]
        # print(doc[start_token.i:end_token.i + 1])
        doc.set_ents([Span(doc, start_token.i, end_token.i + 1, entity["entity_group"])])
        
    return doc

def negation_handling(sentence, entity):
    nlp = spacy.load("en_core_web_sm", disable={"ner"})
    doc = nlp(sentence.lower())
    nlp = medspacy.load(nlp)
    nlp.disable_pipe('medspacy_target_matcher')
    nlp.disable_pipe('medspacy_pyrush')
    doc = nlp(add_custom_entity(doc, entity))
    for e in doc.ents:
        rs = str(e._.is_negated)
        if rs:
            if rs == "True": 
                entity["is_negated"] = "yes"
            elif rs == 'False':
                entity["is_negated"] = "no"
        else:
            entity["is_negated"] = "no"
    return  entity 

class EntityRecognizer:
    def __init__(self, id_list, n_jobs, data_source="clinical trials"):
        self.id_list = id_list
        self.n_jobs = n_jobs
        self.data_source = data_source
        
    def data_loader(self, id_list):
        to_concat = []
        for idx in id_list:
            if self.data_source=="clinical trials":
                df = pd.read_csv(INPUT_FILEPATH + "clinical_trials/" + "%s_preprocessed.csv"%idx)
                to_concat.append(df)
            elif self.data_source=="patient notes":
                df = pd.read_csv(INPUT_FILEPATH + "patient_notes/" + "%s_preprocessed.csv"%idx)
                to_concat.append(df)
            else:
                warnings.warn("Unexpected data source encountered. Please choose between 'clinical trials' or 'patient notes'", UserWarning)
        return to_concat
    
    def mtner_normalize_format(self, json_data):
        spacy_format_entities = []
        for annotation in json_data["annotations"]:
            start = annotation["span"]["begin"]
            end = annotation["span"]["end"]
            label = annotation["obj"]
            mention = annotation["mention"]
            score = annotation["prob"]
            normalized_id = annotation["id"]
            spacy_format_entities.append({
                "entity_group": label,
                "text": mention,
                "score": score,
                "start": start,
                "end": end,
                "normalized_id": normalized_id
            })
        spacy_result = {
            "text": json_data["text"],
            "ents": spacy_format_entities,
        }
        return spacy_result
    
    def merge_lists_with_priority_to_first(self, list1, list2):
        merged_list = list1.copy()  # Create a copy of list1 to preserve its contents
        for dict2 in list2:
            overlap = False
            for dict1 in list1:
                if (dict1['start'] <= dict2['end'] and dict2['start'] <= dict1['start']) or (dict2['start'] <= dict1['end'] and dict1['start'] <= dict2['start']):
                    overlap = True
                    break
            
            if not overlap:
                merged_list.append(dict2)
        return merged_list
    
    def merge_lists_without_priority(self, list1, list2):
        merged_list = list1.copy()  # Create a copy of list1 to preserve its contents
        for dict2 in list2:
            merged_list.append(dict2)
        return merged_list
    
    def find_and_remove_overlaps(self, dictionary_list, if_overlap_keep):
        # Create a dictionary to store non-overlapping entries
        non_overlapping = {}
        # Create a set of entity groups to keep
        preferred_set = set(if_overlap_keep)

        # Iterate through the input list
        for entry in dictionary_list:
            text = entry['text']
            group = entry['entity_group']

            # Check if the text is already in the non_overlapping dictionary
            if text in non_overlapping:
                # Compare groups and keep the entry if it belongs to one of the preferred groups
                if group in preferred_set:
                    non_overlapping[text] = entry
            else:
                non_overlapping[text] = entry

        # Convert the non-overlapping dictionary back to a list
        result_list = list(non_overlapping.values())

        return result_list
    
    def aberration_type_recognizer(self, text):
        med_nlp = medspacy.load()
        med_nlp.disable_pipe('medspacy_target_matcher')
        @Language.component("aberrations-ner")
        def regex_pattern_matcher_for_aberrations(doc):
            df_regex = pd.read_csv("../data/regex_variants.tsv", sep="\t", header=None)
            df_regex = df_regex.rename(columns={1 : "label", 2:"regex_pattern"}).drop(columns=[0])
            dict_regex = df_regex.set_index('label')['regex_pattern'].to_dict()
            original_ents = list(doc.ents)
            # Compile the regex patterns
            compiled_patterns = {
                label: re.compile(pattern)
                for label, pattern in dict_regex.items()
            }
            mwt_ents = []
            for label, pattern in compiled_patterns.items():
                for match in re.finditer(pattern, doc.text):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    if span is not None:
                        mwt_ents.append((label, span.start, span.end, span.text))
                        
            for ent in mwt_ents:
                label, start, end, name = ent
                per_ent = Span(doc, start, end, label=label)
                original_ents.append(per_ent)

            doc.ents = filter_spans(original_ents)
            
            return doc
        med_nlp.add_pipe("aberrations-ner", before='medspacy_context')
        doc = med_nlp(text)
        ent_list =[] 
        for entity in doc.ents:
            ent_list.append({"entity_group" : entity.label_, 
                            "text" : entity.text, 
                            "start": entity.start_char, 
                            "end": entity.end_char, 
                            "is_negated" : "yes" if entity._.is_negated else "no"})
        return ent_list
    
    
    def pregnancy_recognizer(self, text):
        med_nlp = medspacy.load()
        med_nlp.disable_pipe('medspacy_target_matcher')
        
        # Updated regex pattern
        regex_pattern = r"(?i)\b(?:pregn\w+|matern\w+|gestat\w+|lactat\w+|breastfeed\w+|prenat\w+|antenat\w+|postpartum|childbear\w+|parturient|conceiv\w+|obstetr\w+)\b"

        @Language.component("pregnancy-ner")
        def regex_pattern_matcher_for_pregnancy(doc):
            compiled_pattern = re.compile(regex_pattern)

            original_ents = list(doc.ents)
            mwt_ents = []

            for match in re.finditer(compiled_pattern, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is not None:
                    mwt_ents.append((span.start, span.end, span.text))

            for ent in mwt_ents:
                start, end, name = ent
                per_ent = Span(doc, start, end, label="pregnancy")  # Assigning the label "pregnancy"
                original_ents.append(per_ent)

            doc.ents = filter_spans(original_ents)

            return doc

        med_nlp.add_pipe("pregnancy-ner", before='medspacy_context')
        doc = med_nlp(text)
        
        ent_list =[] 
        for entity in doc.ents:
            ent_list.append({
                "entity_group": entity.label_,
                "text": entity.text,
                "start": entity.start_char,
                "end": entity.end_char,
                "is_negated": "yes" if entity._.is_negated else "no"
            })
    
        return ent_list
    
    def merge_similar_consecutive_entities(self, entities):
        combined_entities = []
        if entities:
            current_entity = entities[0]
            for next_entity in entities[1:]:
                if (
                    current_entity['entity_group'] == next_entity['entity_group']
                    and next_entity['start'] - current_entity['end'] - 1 <= 3
                ):
                    current_entity['text'] += ' ' + next_entity['text']
                    current_entity['end'] = next_entity['end']
                else:
                    combined_entities.append(current_entity)
                    current_entity = next_entity

            combined_entities.append(current_entity)
        return combined_entities

    
    def recognize_entities(self, df):
        _ids = []
        sentences = []
        entities_groups = []
        entities_texts = []
        normalized_ids = []
        is_negated = []
        field = []
        start= []
        end = []
        df = df.dropna()
        for _,row in df.iterrows():
            sent = row["sentence"].replace(",", "")
            main_entities = self.mtner_normalize_format(query_plain(sent.lower()))["ents"]
            variants_entities = mutations_pipeline(sent)
            aberration_type_entities = self.aberration_type_recognizer(sent)
            pregnancy_entities = self.pregnancy_recognizer(sent)
            aux_entities = aux_pipeline(sent)
            aux_entities = get_dictionaries_with_values(aux_entities, "entity_group", AUXILIARY_ENTITIES_LIST)
            aux_entities = [{"text" if k == "word" else k: v for k, v in d.items()} for d in aux_entities]
            
            combined_entities  = self.merge_lists_with_priority_to_first(variants_entities, main_entities)
            combined_entities  = self.merge_lists_without_priority(combined_entities, aux_entities)
            combined_entities  = self.merge_lists_without_priority(combined_entities, pregnancy_entities)
            combined_entities  = self.merge_lists_with_priority_to_first(combined_entities, aberration_type_entities)
            combined_entities  = self.merge_similar_consecutive_entities(combined_entities)
            
            # Convert the selected_entries dictionary back to a list
            if len(combined_entities) > 0:
                clean_entities = self.find_and_remove_overlaps(combined_entities, if_overlap_keep=["gene", "ProteinMutation", "DNAMutation", "SNP"])
                for e in clean_entities:
                    if (("score" in e and e["score"] > 0.6) or ("score" not in e)) and len(e["text"]) > 1:
                        ent = negation_handling(sent, e)
                        ent["text"] = re.sub(r'([,.-])\s+', r'\1', e["text"]) 
                        is_negated.append(ent["is_negated"]) 
                        _ids.append(row["id"])
                        sentences.append(sent)
                        entities_groups.append(ent['entity_group'])
                        entities_texts.append(ent['text'])
                        start.append(ent["start"])
                        end.append(ent["end"])
                        if "normalized_id" in ent:
                            normalized_ids.append(ent["normalized_id"])
                        else: 
                            normalized_ids.append("CUI-less")
                        if self.data_source=="clinical trials":
                            field.append(row["criteria"])
                        elif self.data_source=="patient notes":
                            field.append(row["field"])
                    else:
                        continue
        return pd.DataFrame({
                            '_id': _ids,
                            'sentence': sentences,
                            'entity_text': entities_texts,
                            'entity_group': entities_groups,
                            'normalized_id': normalized_ids,
                            'field' : field,
                            "is_negated" : is_negated,
                            'start':start,
                            'end':end
                        })
            
    def __call__(self):
        all_df = self.data_loader(self.id_list)
        parallel_runner = ParallelExecutor(n_jobs=self.n_jobs)(total=len(self.id_list))
        X = parallel_runner(
            joblib.delayed(self.recognize_entities)(
            df, 
            )
            for df in all_df
        )     
        if self.data_source=="clinical_trials":
            pd.concat(X).to_csv(OUTPUT_FILEPATH_CT + "entities_parsed.csv", index = False)
        elif self.data_source=="patient notes":
            pd.concat(X).to_csv(OUTPUT_FILEPATH_PAT + "entities_parsed.csv", index = False)
        return pd.concat(X)
    
    
if __name__ == "__main__":
    main()