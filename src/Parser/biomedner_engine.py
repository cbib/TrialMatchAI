import random
import os
import string
import numpy as np
import hashlib
import time
import asyncio
import socket
import struct
import json
import sys
import traceback
import bioregistry
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import spacy
from io import StringIO
from datetime import datetime
from pathlib import Path
from normalizer import Normalizer
from convert import get_pub_annotation
import gc



# Paths
DICT_PATH = Path("resources/normalization/dictionary")
dict_paths = {
    'gene': DICT_PATH / 'dict_Gene.txt',
    'disease': DICT_PATH / 'dict_Disease_20210630.txt',
    'cell type': DICT_PATH / 'dict_CellType_20210810.txt',
    'drug': DICT_PATH / 'dict_ChemicalCompound_20210630.txt',
    'procedure': DICT_PATH / 'dict_Procedures.txt',
}

class BioMedNER:
    def __init__(self, biomedner_home, biomedner_port, gner_port, gene_norm_port, disease_norm_port,
                 biomedner_host='localhost', gner_host='localhost', time_format='[%d/%b/%Y %H:%M:%S.%f]',
                 max_word_len=100, seed=2019, use_neural_normalizer=True, no_cuda=False):
        self.time_format = time_format
        self.max_word_len = max_word_len
        self.input_dir = os.path.join(biomedner_home, 'input')
        self.output_dir = os.path.join(biomedner_home, 'output')

        self.print_log('BioMedNER LOADING...')
        random.seed(seed)
        np.random.seed(seed)

        self.create_directories([self.input_dir, self.output_dir])
        print(f"Created directories: {self.input_dir}, {self.output_dir}")

        # FOR NER
        self.biomedner_home = biomedner_home
        self.biomedner_host = biomedner_host
        self.biomedner_port = biomedner_port
        
        self.gner_host = gner_host
        self.gner_port = gner_port

        # FOR NEN
        self.normalizer = Normalizer(
            gene_port=gene_norm_port,
            disease_port=disease_norm_port,
            use_neural_normalizer=use_neural_normalizer,
            no_cuda=no_cuda,
            nlp=spacy.load("en_core_web_sm", disable=['ner', 'parser', 'textcat'])
        )
        gc.collect()  # Collect garbage after normalizer initialization
        self.print_log('BioMedNER LOADED...')
        
    def print_log(self, message):
        print(datetime.now().strftime(self.time_format), message)

    def create_directories(self, dirs):
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def delete_files(self, dirname):
        if os.path.exists(dirname):
            for f in os.listdir(dirname):
                f_path = os.path.join(dirname, f)
                if os.path.isfile(f_path):
                    os.remove(f_path)
            self.print_log(f"Deleted files in {dirname}")
            
    def annotate_text(self, text, pmid=None):
        try:
            text = self.preprocess_input(text.strip())
            base_name = self.generate_base_name(text)
            biomed_output, gner_output = self.tag_entities(text, base_name, self.biomedner_home)
            biomed_output = self.process_output(biomed_output, dict_paths)
            gner_output = self.process_output(gner_output, dict_paths)
            output = gner_output + biomed_output
        except Exception as e:
            self.print_log(traceback.format_exc())
            output = {"error_code": 1, "error_message": "Something went wrong. Try again."}

        return output 

    def process_output(self, output, dict_paths):
        if 'annotations' not in output:
            return output
        
        output = self.split_cuis(output)
        output = self.standardize_prefixes(output)
        output = transform_results(output)
        append_synonyms(output, dict_paths)

        return output

    def split_cuis(self, output):
        for anno in output['annotations']:
            cuis = anno['id']
            new_cuis = []
            for cui in cuis:
                if isinstance(cui, list):
                    cui = cui[0]
                new_cuis += cui.replace("|", ",").split(",")
            anno['id'] = new_cuis                 
        return output

    def standardize_prefixes(self, output):
        for anno in output['annotations']:
            cuis = anno['id']
            obj = anno['obj']
            if obj not in ['disease', 'gene', 'drug', 'cell type']:
                continue

            new_cuis = []
            for cui in cuis:
                if "NCBI:txid" in cui: 
                    prefix, numbers = cui.split("NCBI:txid")
                    prefix = "ncbitaxon"
                elif "_" in cui: 
                    prefix, numbers = cui.split("_")
                elif ":" in cui: 
                    prefix, numbers = cui.split(":")
                else:
                    new_cuis.append(cui)
                    continue
                    
                prefix = bioregistry.normalize_prefix(prefix) or prefix
                prefix = bioregistry.get_preferred_prefix(prefix) or prefix

                if prefix == 'cellosaurus':
                    numbers = "CVCL_" + numbers

                new_cuis.append(":".join([prefix, numbers]))
            
            anno['id'] = new_cuis

        return output

    def preprocess_input(self, text):
        replacements = {
            '\r\n': ' ', '\n': ' ', '\t': ' ', '\xa0': ' ', '\x0b': ' ', '\x0c': ' '
        }
        for old, new in replacements.items():
            if old in text:
                self.print_log(f'Found {old} -> replace it w/ a space')
                text = text.replace(old, new)
        
        text = text.encode("ascii", "ignore").decode()

        tokens = text.split(' ')
        for idx, tk in enumerate(tokens):
            if len(tk) > self.max_word_len:
                tokens[idx] = tk[:self.max_word_len]
                self.print_log('Found a too long word -> cut the suffix of the word')
        text = ' '.join(tokens)

        return text

    def tag_entities(self, text, base_name, biomedner_home):
        n_ascii_letters = sum(1 for l in text if l in string.ascii_letters)
        if n_ascii_letters == 0:
            return 'No ascii letters. Please enter your text in English.', ''

        pubtator_file = f'{base_name}.PubTator'
        input_data = f'{base_name}|t|\n{base_name}|a|{text}\n\n'

        input_biomedner = os.path.join(self.input_dir, f'{pubtator_file}.biomedner.PubTator')
        output_biomedner = os.path.join(self.output_dir, f'{pubtator_file}.biomedner.json')
        input_gner = os.path.join(self.input_dir, f'{pubtator_file}.gner.PubTator')
        output_gner = os.path.join(self.output_dir, f'{pubtator_file}.gner.json')

        self.create_directories([self.input_dir, self.output_dir])

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(self.write_to_file, path, input_data): path 
                for path in [input_biomedner, input_gner]
            }
            wait(future_to_file, return_when=ALL_COMPLETED)

        ner_start_time = time.time()

        async_result = asyncio.run(self.async_ner([
            ('biomedner', pubtator_file, output_biomedner, biomedner_home),
            ('gner', pubtator_file, output_gner, biomedner_home)
        ]))

        ner_elapse_time = time.time() - ner_start_time
        async_result['ner_elapse_time'] = ner_elapse_time

        tagged_docs = async_result['biomedner_tagged_docs']
        gner_entities = async_result['gner_tagged_docs']

        r_norm_start_time = time.time()
        tagged_docs = self.normalize_entities(tagged_docs, async_result['biomedner_num_entities'], base_name)
        gner_entities = self.normalize_entities(gner_entities, async_result['gner_num_entities'], base_name)
        r_norm_elapse_time = time.time() - r_norm_start_time
        async_result['r_norm_elapse_time'] = r_norm_elapse_time

        n_norm_start_time = time.time()
        if self.normalizer.use_neural_normalizer:
            if async_result['biomedner_num_entities'] > 0:
                tagged_docs = self.normalizer.neural_normalize(ent_type='disease', tagged_docs=tagged_docs)
                tagged_docs = self.normalizer.neural_normalize(ent_type='drug', tagged_docs=tagged_docs)
                tagged_docs = self.normalizer.neural_normalize(ent_type='gene', tagged_docs=tagged_docs)
        n_norm_elapse_time = time.time() - n_norm_start_time
        async_result['n_norm_elapse_time'] = n_norm_elapse_time

        tagged_docs[0] = get_pub_annotation(tagged_docs[0])
        tagged_docs[0]['elapse_time'] = self.get_elapsed_times(async_result)
        
        gner_entities[0] = get_pub_annotation(gner_entities[0])
        gner_entities[0]['elapse_time'] = {'gner_elapse_time': async_result['gner_elapse_time']}

        self.cleanup_temp_files([input_biomedner, output_biomedner, input_gner, output_gner])
        gc.collect()
        return tagged_docs[0], gner_entities[0]

    def get_elapsed_times(self, async_result):
        return {
            'biomedner_elapse_time': async_result['biomedner_elapse_time'],
            'gner_elapse_time': async_result['gner_elapse_time'],
            'ner_elapse_time': async_result['ner_elapse_time'],
            'r_norm_elapse_time': async_result['r_norm_elapse_time'],
            'n_norm_elapse_time': async_result['n_norm_elapse_time'],
            'norm_elapse_time': async_result['r_norm_elapse_time'] + async_result['n_norm_elapse_time'],
        }

    def normalize_entities(self, entities, num_entities, base_name):
        if num_entities > 0:
            entities = self.normalizer.normalize(base_name, entities)
        return entities

    def write_to_file(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)

    def generate_base_name(self, text):
        return hashlib.sha224((text + str(time.time())).encode('utf-8')).hexdigest()

    async def async_ner(self, arguments):
        coroutines = [self._ner_wrap(arg) for arg in arguments]
        result = await asyncio.gather(*coroutines)
        return {k: v for e in result for k, v in e.items()}

    async def _ner_wrap(self, ner_type_info):
        ner_type, pubtator_file, output_file, biomedner_home = ner_type_info
        retries = 1

        for _ in range(retries):
            try:
                start_time = time.time()
                await async_tell_inputfile(
                    self.biomedner_host,
                    self.biomedner_port if ner_type == 'biomedner' else self.gner_port,
                    biomedner_home,
                    pubtator_file,
                    asyncio.get_event_loop()
                )

                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    try:
                        tagged_docs = [json.loads(content)]
                    except json.JSONDecodeError as e:
                        self.print_log(f"JSON decode error: {e}")
                        continue

                num_entities = tagged_docs[0]['num_entities']
                if tagged_docs is None:
                    return None

                elapse_time = time.time() - start_time
                return {f"{ner_type}_elapse_time": elapse_time, f"{ner_type}_tagged_docs": tagged_docs, f"{ner_type}_num_entities": num_entities}

            except json.JSONDecodeError as e:
                self.print_log(f"JSON decode error: {e}")
                time.sleep(0)

        raise Exception("Failed to decode JSON after multiple attempts")

    def cleanup_temp_files(self, files):
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def annotate_single_text_with_retry(self, text, retries=1, delay=0):
        for attempt in range(retries):
            try:
                return self.annotate_text(text)
            except (ConnectionResetError, ConnectionRefusedError) as e:
                self.print_log(f"Error: {e}. Retrying {attempt + 1}/{retries} in {delay} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to annotate text after {retries} attempts: {text}")

    def annotate_texts_in_parallel(self, texts, max_workers=100, retries=1, delay=0):
        results = [None] * len(texts)  
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.annotate_single_text_with_retry, text, retries, delay): i for i, text in enumerate(texts)}
            while future_to_index:
                done, _ = wait(future_to_index, return_when=ALL_COMPLETED)
                for future in done:
                    index = future_to_index.pop(future)
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        self.print_log(f'Text at index {index} generated an exception: {exc}')
                        results[index] = {"error_code": 1, "error_message": str(exc)}
        gc.collect()
        return results

async def async_tell_inputfile(host, port, biomedner_home, inputfile, loop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        message = json.dumps({"biomedner_home": biomedner_home, "inputfile": inputfile})
        input_stream = struct.pack('>H', len(message)) + message.encode('utf-8')
        sock.send(input_stream)
        output_stream = await loop.run_in_executor(None, sock.recv, 512)
        resp = output_stream.decode('utf-8')
        sock.close()
        return resp
    except (ConnectionRefusedError, TimeoutError, ConnectionResetError) as e:
        print(e)
        return None

def get_synonyms_from_file(file_path, entity_ids):
    entity_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('||')
            if len(parts) > 1:
                ids = parts[0].split(',')
                names = parts[1].split('|')
                for nid in ids:
                    entity_dict[nid.strip()] = names

    synonyms = []
    for nid in entity_ids:
        synonyms.extend(entity_dict.get(nid.upper(), []))
    return synonyms

def append_synonyms(ner_results, dict_paths):
    valid_entity_groups = ['disease', 'gene', 'drug', 'cell type', 'diagnostic procedure', 'therapeutic procedure', 'laboratory procedure']
    dict_files = {k: v for k, v in dict_paths.items() if k in valid_entity_groups}

    for entity in ner_results:
        entity_group = entity['entity_group'].lower()
        if entity_group in dict_files:
            file_path = dict_files[entity_group]
            if entity_group == 'gene':
                entity['normalized_id'] = [nid.replace('NCBIGene:', 'EntrezGene:') for nid in entity['normalized_id']]
                entity['synonyms'] = get_synonyms_from_file(file_path, entity['normalized_id'])
            else:
                entity['synonyms'] = get_synonyms_from_file(file_path, entity['normalized_id'])
        else:
            entity['synonyms'] = []
        
def transform_results(data):
    all_entities = []

    for annotation in data['annotations']:
        if annotation['prob'] > 0.5:
            entity = {
                'entity_group': annotation['obj'], 
                'score': annotation['prob'],  
                'text': annotation['mention'], 
                'start': annotation['span']['begin'],  
                'end': annotation['span']['end'],  
                'normalized_id': annotation['id'],  
            }
            all_entities.append(entity)

    all_entities.sort(key=lambda x: (x['start'], x['end'], -x['score']))

    non_overlapping_entities = []
    last_end = -1

    for entity in all_entities:
        if entity['start'] >= last_end:
            non_overlapping_entities.append(entity)
            last_end = entity['end']

    return non_overlapping_entities

# def resolve_overlaps(entities, priority_groups):
#     df = pd.read_csv("../../data/regex_variants.tsv", sep="\t", header=None)
#     variants_list = df[0].values.tolist()
#     variants_list.extend(priority_groups)
#     priority_groups = variants_list

#     entities_sorted = sorted(entities, key=lambda x: (x['start'], -x['end']))
#     accepted_entities = []

#     for current in entities_sorted:
#         overlap = False
#         for i, accepted in enumerate(list(accepted_entities)): 
#             if (accepted['start'] < current['end'] and current['start'] < accepted['end']):
#                 overlap = True
#                 if accepted['entity_group'] not in priority_groups:
#                     if current['entity_group'] in priority_groups:
#                         accepted_entities[i] = current
#                 elif current['entity_group'] in priority_groups:
#                     accepted_entities[i] = current
#                 break
#         if not overlap:
#             accepted_entities.append(current)
            
#     return accepted_entities

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_word_len', type=int, help='word max chars', default=50)
    argparser.add_argument('--seed', type=int, help='seed value', default=2019)
    argparser.add_argument('--biomedner_home', help='biomedical language model home')
    argparser.add_argument('--biomedner_host', help='biomedical language model host', default='localhost')
    argparser.add_argument('--biomedner_port', type=int, help='biomedical language model port', default=18894)
    argparser.add_argument('--gner_host', help='gner host', default='localhost')
    argparser.add_argument('--gner_port', type=int, help='gner port', default=18783)
    argparser.add_argument('--gene_norm_port', type=int, help='Gene port', default=18888)
    argparser.add_argument('--disease_norm_port', type=int, help='Sieve port', default=18892)
    argparser.add_argument('--time_format', help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
    argparser.add_argument("--use_neural_normalizer", action="store_true")
    argparser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()

    biomedner = BioMedNER(
        max_word_len=args.max_word_len,
        seed=args.seed,
        gene_norm_port=args.gene_norm_port,
        disease_norm_port=args.disease_norm_port,
        biomedner_home=args.biomedner_home,
        biomedner_host=args.biomedner_host,
        biomedner_port=args.biomedner_port,
        gner_host=args.gner_host,
        gner_port=args.gner_port,
        time_format=args.time_format,
        use_neural_normalizer=args.use_neural_normalizer,
        no_cuda=args.no_cuda,
    )

    texts = ["Colorectal Cancer", "Fallopian Tube Cancer", "Carcinoma"]
    results = biomedner.annotate_texts_in_parallel(texts, max_workers=5)
    for result in results:
        print(result)
