# server_script.py
import os
import json
import socket
import struct
import argparse
import warnings
import re
import pandas as pd
from gliner import GLiNER
from ops import pubtator2dict_list
import logging
from concurrent.futures import ThreadPoolExecutor


warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def entities_highly_overlap(entity1, entity2, threshold=0.5):
    start1, end1 = entity1['start'], entity1['end']
    start2, end2 = entity2['start'], entity2['end']
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    length1 = end1 - start1
    length2 = end2 - start2
    shorter_length = min(length1, length2)
    if shorter_length == 0:
        return False
    overlap_ratio = overlap_length / shorter_length
    return overlap_ratio >= threshold

def filter_overlapping_entities(entities, overlap_threshold=0.5):
    # Sort entities by score descending
    entities = sorted(entities, key=lambda x: x.get('score', 0), reverse=True)
    selected_entities = []
    for entity in entities:
        overlaps = False
        for sel_entity in selected_entities:
            if entities_highly_overlap(entity, sel_entity, threshold=overlap_threshold):
                overlaps = True
                break
        if not overlaps:
            selected_entities.append(entity)
    return selected_entities

class GNER:
    def __init__(self, params):
        self.params = params
        self.entities = [
            'diagnostic test', 'treatment', 'laboratory test', 'surgical procedure',
            'sign symptom', 'radiology', 'genomic analysis technique'
        ]
        self.model = GLiNER.from_pretrained(self.params.model_name_or_path, map_location='cpu')

    def recognize(self, text, base_name):
        entities = self.model.predict_entities(text, self.entities, flat_ner=False, threshold=0.8)

        clean_entities = [{
            "entity_group": d["label"],
            "score": float(d.get("score", 0)),
            "text": d["text"],
            "start": int(d["start"]),
            "end": int(d["end"])
        } for d in entities]

        # clean_entities.extend(aberration_entities)

        # Filter overlapping entities
        filtered_entities = filter_overlapping_entities(clean_entities, overlap_threshold=0.75)

        entity_dict = {entity: [] for entity in self.entities}
        prob_dict = {entity: [] for entity in self.entities}

        for entity in filtered_entities:
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
            "num_entities": len(filtered_entities)
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
        s.listen(600)
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
