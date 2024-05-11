import socket
import struct
import json
import argparse
import torch
from transformers import AutoTokenizer, pipeline
import warnings

import re
import medspacy
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def get_dictionaries_of_specific_entities(list_of_dicts, key, values):
    return [d for d in list_of_dicts if any(val in d.get(key, []) for val in values)]

class BioMedNERMacrobbat:
    def __init__(self, params):
        self.params = params
        self.entities = ["Sign_symptom", "Biological_structure", "Date", "Duration", "Time", "Frequency", 
                         "Severity", "Lab_value", "Dosage", "Diagnostic_procedure", "Therapeutic_procedure", "Medication", 
                         "Clinical_event", "Outcome", "History", "Subject", "Family_history", "Detailed_description", "Area"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.model_path_or_name, model_max_length=512, truncation=True)
        self.ner_pipeline = pipeline("ner", model=self.params.model_path_or_name, tokenizer=self.tokenizer, aggregation_strategy="first", device=self.params.device)

    def recognize(self, text):
        entities = self.ner_pipeline(text)
        entities = get_dictionaries_of_specific_entities(entities, "entity_group", self.entities)
        pregnancy_entities = pregnancy_recognizer(text)
        aberration_entities = aberration_type_recognizer(text)
        entities.extend(pregnancy_entities)
        entities.extend(aberration_entities)
        # Ensure all data is JSON serializable
        clean_entities = [{
            "entity_group": d["entity_group"],
            "score": float(d["score"]) if "score" in d else None,  # Convert numpy float32 to Python float
            "text": d["word"],
            "start": int(d["start"]),  # Ensure start and end are ints
            "end": int(d["end"])
        } for d in entities]
        return clean_entities
    

def pregnancy_recognizer(text):
    med_nlp = medspacy.load()
    med_nlp.disable_pipe('medspacy_target_matcher')
    regex_pattern = r"(?ix)\b(?:pregn\w*|matern\w*|gestat\w*|lactat\w*|breastfeed\w*|prenat\w*|antenat\w*|postpartum|childbear\w*|parturient|conceiv\w*|obstetr\w*|fertil\w*|gravid\w*|perinat\w*|neonat\w*|postnatal|childbirth|delivery|birthing|expectant\ mother|nursing\ mother|puerperal|midwifery|reproductive\ health|expecting(\ a\ child|\ baby)?)\b"
    
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
            per_ent = Span(doc, start, end, label="pregnancy")  # Assigning the label "PREGNANCY"
            original_ents.append(per_ent)
        
        doc.ents = filter_spans(original_ents)
        return doc

    # Add the component to the pipeline
    med_nlp.add_pipe("pregnancy-ner")
    
    # Process the input text with the modified pipeline
    doc = med_nlp(text)
    ent_list =[] 
    for entity in doc.ents:
        ent_list.append({"entity_group" : entity.label_, 
                        "word" : entity.text, 
                        "start": entity.start_char, 
                        "end": entity.end_char})
    return ent_list

def aberration_type_recognizer(text):
    med_nlp = medspacy.load()
    med_nlp.disable_pipe('medspacy_target_matcher')
    @Language.component("aberrations-ner")
    def regex_pattern_matcher_for_aberrations(doc):
        df_regex = pd.read_csv("../../data/regex_variants.tsv", sep="\t", header=None)
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
                        "word" : entity.text, 
                        "start": entity.start_char, 
                        "end": entity.end_char})
    return ent_list


def handle_client_connection(connection, model):
    while True:
        # Receive data from the client
        data_length = connection.recv(2)
        if not data_length:
            break
        data_length = struct.unpack('>H', data_length)[0]
        text = connection.recv(data_length).decode('utf-8')

        # Process the text using the model
        entities = model.recognize(text)

        # Send back the results
        response = json.dumps(entities)
        connection.send(struct.pack('>H', len(response)) + response.encode('utf-8'))

    connection.close()

def run_server(model, args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((args.host, args.port))
        server_socket.listen(5)
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, _ = server_socket.accept()
            handle_client_connection(conn, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioMedNERMacrobbat Server")
    parser.add_argument("--model_path_or_name", type=str, help="Path to the model")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()
    
    ner_model = BioMedNERMacrobbat(args)
    run_server(ner_model, args)
