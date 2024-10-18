# server_script.py
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
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_aberration_labels():
    df_regex = pd.read_csv("../../data/regex/regex_variants.tsv", sep="\t", header=None)
    df_regex = df_regex.rename(columns={1: "label", 2: "regex_pattern"}).drop(columns=[0])
    return df_regex.set_index('label')['regex_pattern'].to_dict()

def genomic_aberration_type_recognizer(text, compiled_patterns):
    med_nlp = medspacy.load()
    med_nlp.disable_pipe('medspacy_target_matcher')

    @Language.component("aberrations-ner")
    def regex_pattern_matcher_for_aberrations(doc):
        original_ents = list(doc.ents)
        mwt_ents = [
            (label, *match.span(), match.group())
            for label, pattern in compiled_patterns.items()
            for match in re.finditer(pattern, doc.text)
        ]
        for label, start, end, name in mwt_ents:
            span = doc.char_span(start, end, label=label)
            if span:
                original_ents.append(span)
        doc.ents = filter_spans(original_ents)
        return doc

    compiled_patterns = {label: re.compile(pattern) for label, pattern in compiled_patterns.items()}
    med_nlp.add_pipe("aberrations-ner", before='medspacy_context')
    doc = med_nlp(text)

    return [{
        "entity_group": entity.label_,
        "text": entity.text,
        "start": entity.start_char,
        "end": entity.end_char,
        "score": None
    } for entity in doc.ents]

class GNER:
    def __init__(self, params):
        self.params = params
        self.entities = [
            "medication", "sign symptom", "duration", "disease status", "performance status", "laboratory test value",
            "laboratory procedure", "severity", "genetic mutation", "mutation status", "diagnostic procedure", 
            "therapeutic procedure", "procedure outcome", "pregnancy status"
        ]
        self.model = GLiNER.from_pretrained(self.params.model_name_or_path, map_location='cuda')
        self.compiled_patterns = {label: re.compile(pattern) for label, pattern in load_aberration_labels().items()}
   
    def recognize(self, text, base_name):
        entities = self.model.predict_entities(text, self.entities, multi_label=True, flat_ner=False, threshold=0.6)
        aberration_entities = genomic_aberration_type_recognizer(text, self.compiled_patterns)
        
        clean_entities = [{
            "entity_group": d["label"],
            "score": float(d.get("score", 0)),
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

def gner_recognize(model, input_file, output_file, base_name):
    try:
        dict_list = pubtator2dict_list(input_file)
        texts = [entry['abstract'] for entry in dict_list]

        results = [model.recognize(text, base_name) for text in texts]

        if not results:
            return None, 0

        num_entities = sum(r['num_entities'] for r in results)
        results[0]['num_entities'] = num_entities  # Total number of entities across all texts

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results[0], f)

        return results, num_entities
    except Exception as e:
        logging.error(f"An error occurred in gner_recognize: {e}")
        return None, 0

def handle_connection(conn, addr, model, args):
    try:
        with conn:
            logging.info(f"Connection accepted from {addr}")
            message_length = struct.unpack('>H', conn.recv(2))[0]
            message = conn.recv(message_length).decode('utf-8').replace('\x00', '')
            logging.info(f"Received data: {message}")
            data = json.loads(message)
            biomedner_home = data['biomedner_home']
            inputfile = data['inputfile']
            base_name = os.path.splitext(os.path.basename(inputfile))[0]
            
            input_file = os.path.join(biomedner_home, 'input', f'{inputfile}.gner.PubTator')
            output_file = os.path.join(biomedner_home, 'output', f'{inputfile}.gner.json')

            gner_recognize(model, input_file, output_file, base_name)
            
            output_stream = struct.pack('>H', len(inputfile)) + inputfile.encode('utf-8')
            conn.send(output_stream)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def run_server(model, args):
    host = args.gner_host
    port = args.gner_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(300)
        logging.info(f"Server started on {host}:{port}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            while True:
                try:
                    conn, addr = s.accept()
                    executor.submit(handle_connection, conn, addr, model, args)
                except Exception as e:
                    logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    argparser.add_argument('--gner_host', default='localhost', help='biomedical language model host')
    argparser.add_argument('--gner_port', type=int, default=18894, help='biomedical language model port')
    argparser.add_argument('--time_format', default='[%d/%b/%Y %H:%M:%S.%f]', help='time format')
    args = argparser.parse_args()
    
    mt_ner = GNER(args)  
    run_server(mt_ner, args)
