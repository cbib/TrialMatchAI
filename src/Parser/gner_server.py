import os
import json
import socket
import struct
import argparse
import warnings
import re
import medspacy
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language
import pandas as pd
from gliner import GLiNER
from ops import pubtator2dict_list
import logging

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        for ent_type, entities in d['entities'].items():
            num_entities += len(entities)
    return num_entities

def load_aberration_labels():
    df_regex = pd.read_csv("../../data/regex_variants.tsv", sep="\t", header=None)
    df_regex = df_regex.rename(columns={1: "label", 2: "regex_pattern"}).drop(columns=[0])
    return df_regex['label'].unique().tolist()

def genomic_aberration_type_recognizer(text):
    med_nlp = medspacy.load()
    med_nlp.disable_pipe('medspacy_target_matcher')
    @Language.component("aberrations-ner")
    def regex_pattern_matcher_for_aberrations(doc):
        df_regex = pd.read_csv("../../data/regex_variants.tsv", sep="\t", header=None)
        df_regex = df_regex.rename(columns={1: "label", 2: "regex_pattern"}).drop(columns=[0])
        dict_regex = df_regex.set_index('label')['regex_pattern'].to_dict()
        original_ents = list(doc.ents)
        compiled_patterns = {label: re.compile(pattern) for label, pattern in dict_regex.items()}
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
    ent_list = []
    for entity in doc.ents:
        ent_list.append({"entity_group": entity.label_, "text": entity.text, "start": entity.start_char, "end": entity.end_char, "score": None})
    return ent_list

class GNER:
    def __init__(self, params):
        self.params = params
        self.entities = ["disease", "gene", "cell type", "drug", "medication", "sign symptom", "time", "duration", 
                         "disease status", "performance status", "laboratory measurement", "laboratory procedure","severity", 
                         "DNA mutation", "mutation status", "diagnostic procedure",  "therapeutic procedure", 
                         "treatment outcome", "pregnancy status", "age"]
        self.model = GLiNER.from_pretrained(self.params.model_name_or_path)
   
    def recognize(self, text, base_name):
        entities = self.model.predict_entities(text, self.entities, multi_label=True, flat_ner=False, threshold=0.6)
        aberration_entities = genomic_aberration_type_recognizer(text)
        clean_entities = [{
            "entity_group": d["label"],
            "score": float(d["score"]) if "score" in d else None,
            "text": d["text"],
            "start": int(d["start"]),
            "end": int(d["end"])
        } for d in entities]
        clean_entities.extend(aberration_entities)

        entity_dict = {entity: [] for entity in self.entities}
        prob_dict = {entity: [] for entity in self.entities}

        for entity in clean_entities:
            entity_type = entity["entity_group"]
            if entity_type in entity_dict:
                entity_dict[entity_type].append({"start": entity["start"], "end": entity["end"]})
                prob_dict[entity_type].append([{"start": entity["start"], "end": entity["end"]}, entity["score"]])

        return {
            "pmid": base_name,
            "entities": entity_dict,
            "title": "",  
            "abstract": text,
            "prob": prob_dict,
            "num_entities": len(clean_entities)
        }

def gner_recognize(model, dict_path, base_name, args):
    input_mt_ner = os.path.join(args.gner_home, 'input', f'{dict_path[2:]}.PubTator')
    output_mt_ner = os.path.join(args.gner_home, 'output', f'{dict_path[2:]}.json')

    try:
        dict_list = pubtator2dict_list(input_mt_ner)
        print(f"dict_list: {dict_list}")
        texts = [entry['abstract'] for entry in dict_list]
        
        results = []
        for text in texts:
            result = model.recognize(text, base_name)
            results.append(result)

        if not results:
            return None, 0

        num_entities = sum(r['num_entities'] for r in results)
        results[0]['num_entities'] = num_entities  # Total number of entities across all texts

        # Write output to a .json file
        with open(output_mt_ner, 'w', encoding='utf-8') as f:
            json.dump(results[0], f)

        # return results, num_entities
    except Exception as e:
        logging.error(f"An error occurred in gner_recognize: {e}")
        return None, 0


logging.basicConfig(level=logging.DEBUG)

def run_server(model, args):
    host = args.gner_host
    port = args.gner_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        logging.info(f"Server started on {host}:{port}")
        while True:
            try:
                conn, addr = s.accept()
                logging.info(f"Connection accepted from {addr}")
                with conn:
                    dict_path = conn.recv(512).decode('utf-8')
                    logging.info(f"Received data: {dict_path}")
                    base_name = dict_path.split('.')[0]
                    base_name = base_name.replace("\x00A", "")
                    
                    gner_recognize(model, dict_path, base_name, args)
                    
                    output_stream = struct.pack('>H', len(dict_path)) + dict_path.encode(
                        'utf-8')

                    conn.send(output_stream)
                    conn.close()
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, help='Path to the model')
    argparser.add_argument('--gner_home', help='biomedical language model home')         
    argparser.add_argument('--gner_host', help='biomedical language model host', default='localhost')
    argparser.add_argument('--gner_port', type=int, help='biomedical language model port', default=18894)
    argparser.add_argument('--time_format', help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
    args = argparser.parse_args()
    
    mt_ner = GNER(args)  
    run_server(mt_ner, args)
