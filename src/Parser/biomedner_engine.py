import random
import requests
import os
import string
import numpy as np
import hashlib
import time
import shutil
import asyncio
import socket
import struct
import json
import sys
from datetime import datetime
from collections import OrderedDict
import traceback
import bioregistry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import pubtator2dict_list, get_pub_annotation
from normalizer import Normalizer

# Example usage:
DICT_PATH = "resources/normalization/dictionary"
dict_paths = {
    'gene': os.path.join(DICT_PATH, 'dict_Gene.txt'),
    'disease': os.path.join(DICT_PATH, 'dict_Disease_20210630.txt'),
    'cell_line': os.path.join(DICT_PATH, 'dict_CellLine_20210520.txt'),
    'cell_type': os.path.join(DICT_PATH, 'dict_CellType_20210810.txt'),
    'drug': os.path.join(DICT_PATH, 'dict_ChemicalCompound_20210630.txt'),
    'species': os.path.join(DICT_PATH, 'dict_Species.txt')
}
class BioMedNER():
    def __init__(self, 
        gnormplus_home,
        gnormplus_port,
        biomedner_home,
        biomedner_port,
        maccrobat_port,
        gene_norm_port,
        disease_norm_port,
        gnormplus_host='localhost',
        biomedner_host='localhost',
        maccrobat_host='localhost',
        time_format='[%d/%b/%Y %H:%M:%S.%f]',
        max_word_len=50, 
        seed=2019,
        use_neural_normalizer=True,
        keep_files=False,
        no_cuda=False):

        self.time_format = time_format

        print(datetime.now().strftime(self.time_format), 'BioMedNER LOADING...')
        random.seed(seed)
        np.random.seed(seed)

        if not os.path.exists('./output'):
            os.mkdir('output')

        # delete prev. version outputs
        if not keep_files:
            delete_files('./output')
            delete_files(os.path.join(gnormplus_home, 'input'))
            delete_files(os.path.join('./multi_ner', 'input'))
            delete_files(os.path.join('./multi_ner', 'tmp'))
            delete_files(os.path.join('./multi_ner', 'output'))

        # FOR NER
        self.gnormplus_home =  gnormplus_home
        self.gnormplus_host = gnormplus_host
        self.gnormplus_port = gnormplus_port

        self.biomedner_home =  biomedner_home
        self.biomedner_host = biomedner_host
        self.biomedner_port = biomedner_port
        
        self.maccrobat_host = maccrobat_host
        self.maccrobat_port = maccrobat_port

        self.max_word_len = max_word_len

        # FOR NEN
        self.normalizer = Normalizer(
            gene_port = gene_norm_port,
            disease_port = disease_norm_port,
            use_neural_normalizer = use_neural_normalizer,
            no_cuda = no_cuda
        )

        print(datetime.now().strftime(self.time_format), 'BioMedNER LOADED...')
    
    def annotate_text(self, text, pmid=None):
        try:
            text = text.strip()
            base_name = self.generate_base_name(text) # for the name of temporary files
            text = self.preprocess_input(text, base_name)
            biomed_output, maccrobat_output = self.tag_entities(text, base_name)
            biomed_output['error_code'], biomed_output['error_message'] = 0, ""
            biomed_output = self.post_process_output(biomed_output)
            biomed_output = transform_results(biomed_output)
            append_synonyms(biomed_output, dict_paths)
            maccrobat_output.extend(biomed_output)
            output = maccrobat_output
        except Exception as e:
            errStr = traceback.format_exc()
            print(errStr)

            output = {"error_code": 1, "error_message": "Something went wrong. Try again."}

        return output

    def post_process_output(self, output):
        # hotfix
        if 'annotations' not in output:
            return output
        
        # split_cuis (e.g., "OMIM:608627,MESH:C563895" => ["OMIM:608627","MESH:C563895"])
        output = self.split_cuis(output)

        # standardize prefixes (e.g., EntrezGene:10533 => NCBIGene:10533)
        output = self.standardize_prefixes(output)

        return output

    def split_cuis(self, output):
        # "OMIM:608627,MESH:C563895" or "OMIM:608627|MESH:C563895" 
        # => ["OMIM:608627","MESH:C563895"]

        for anno in output['annotations']:
            cuis = anno['id']
            new_cuis = []
            for cui in cuis:
                # hotfix in case cui is ['cui-less']
                if isinstance(cui, list):
                    cui = cui[0]

                new_cuis += cui.replace("|", ",").split(",")
            anno['id'] = new_cuis                 
        return output

    def standardize_prefixes(self, output):
        # EntrezGene:10533 => NCBIGene:10533
        for anno in output['annotations']:
            cuis = anno['id']
            obj = anno['obj']
            if obj not in ['disease', 'gene', 'drug', 'species', 'cell_line', 'cell_type']:
                continue

            new_cuis = []
            for cui in cuis:
                if "NCBI:txid" in cui: # NCBI:txid10095
                    prefix, numbers = cui.split("NCBI:txid")
                    prefix = "ncbitaxon"
                elif "_" in cui: # CVCL_J260
                    prefix, numbers = cui.split("_")
                elif ":" in cui: # MESH:C563895
                    prefix, numbers = cui.split(":")
                else:
                    new_cuis.append(cui)
                    continue
                    
                normalized_prefix = bioregistry.normalize_prefix(prefix)
                if normalized_prefix is not None:
                    prefix = normalized_prefix
                
                preferred_prefix = bioregistry.get_preferred_prefix(prefix)
                if preferred_prefix is not None:
                    prefix = preferred_prefix
                
                # to convert CVCL_J260 to cellosaurus:CVCL_J260
                if prefix == 'cellosaurus':
                    numbers = "CVCL_" + numbers

                new_cuis.append(":".join([prefix,numbers]))
            
            anno['id'] = new_cuis

        return output

    
    def preprocess_input(self, text, base_name):
        if '\r\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a CRLF -> replace it w/ a space')
            text = text.replace('\r\n', ' ')

        if '\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a line break -> replace it w/ a space')
            text = text.replace('\n', ' ')

        if '\t' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a tab -> replace w/ a space')
            text = text.replace('\t', ' ')

        if '\xa0' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\xa0 -> replace w/ a space')
            text = text.replace('\xa0', ' ')

        if '\x0b' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\x0b -> replace w/ a space')
            text = text.replace('\x0b', ' ')
            
        if '\x0c' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\x0c -> replace w/ a space')
            text = text.replace('\x0c', ' ')
        
        # remove non-ascii
        text = text.encode("ascii", "ignore").decode()

        found_too_long_words = 0
        tokens = text.split(' ')
        for idx, tk in enumerate(tokens):
            if len(tk) > self.max_word_len:
                tokens[idx] = tk[:self.max_word_len]
                found_too_long_words += 1
        if found_too_long_words > 0:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a too long word -> cut the suffix of the word')
            text = ' '.join(tokens)

        return text

    def tag_entities(self, text, base_name):
        n_ascii_letters = 0
        for l in text:
            if l not in string.ascii_letters:
                continue
            n_ascii_letters += 1

        if n_ascii_letters == 0:
            text = 'No ascii letters. Please enter your text in English.'

        base_name = self.generate_base_name(text)
        print(datetime.now().strftime(self.time_format),
              f'id: {base_name}')

        pubtator_file = f'{base_name}.PubTator'
        input_gnormplus = os.path.join(self.gnormplus_home, 'input', pubtator_file)
        output_gnormplus = os.path.join(self.gnormplus_home, 'output', pubtator_file)

        input_biomedner = os.path.join(self.biomedner_home, 'input',
                                     f'{pubtator_file}.PubTator')
        output_biomedner = os.path.join(self.biomedner_home, 'output',
                                     f'{pubtator_file}.json')

        if not os.path.exists(self.biomedner_home + '/input'):
            os.mkdir(self.biomedner_home + '/input')
        if not os.path.exists(self.biomedner_home + '/output'):
            os.mkdir(self.biomedner_home + '/output')

        # Write input str to a .PubTator format file
        with open(input_gnormplus, 'w', encoding='utf-8') as f:
            # only abstract
            f.write(f'{base_name}|t|\n')
            f.write(f'{base_name}|a|{text}\n\n')

        shutil.copy(input_gnormplus, input_biomedner)
        ner_start_time = time.time()
        
        # async call for gnormplus 
        arguments_for_coroutines = []
        loop = asyncio.new_event_loop()
        for ner_type in ['gnormplus', 'biomedner', 'maccrobat']:
            arguments_for_coroutines.append([ner_type, pubtator_file, output_biomedner, base_name, loop])
        async_result = loop.run_until_complete(self.async_ner(arguments_for_coroutines))
        loop.close()
        gnormplus_elapse_time = async_result['gnormplus_elapse_time']
        biomedner_elapse_time = async_result['biomedner_elapse_time']
        maccrobat_elapse_time = async_result['maccrobat_elapse_time']
        # get output result to merge
        tagged_docs = async_result['tagged_docs']
        num_entities = async_result['num_entities']
        maccrobat_entities = async_result['maccrobat_resp']
        
        ner_elapse_time = time.time() - ner_start_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NER {ner_elapse_time} sec')

        # Rule-based Normalization models
        r_norm_start_time = time.time()
        if num_entities > 0:
            tagged_docs = self.normalizer.normalize(base_name, tagged_docs)
        r_norm_elapse_time = time.time() - r_norm_start_time

        # Neural-based normalization models
        n_norm_start_time = time.time()
        if self.normalizer.use_neural_normalizer and num_entities > 0:
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='disease', 
                tagged_docs=tagged_docs
            )
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='drug', 
                tagged_docs=tagged_docs
            )
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='gene', 
                tagged_docs=tagged_docs
            )


        n_norm_elapse_time = time.time() - n_norm_start_time

        print(datetime.now().strftime(self.time_format),
            f'[{base_name}] Neural Normalization {n_norm_elapse_time} sec')

        # Convert to PubAnnotation JSON
        tagged_docs[0] = get_pub_annotation(tagged_docs[0])

        norm_elapse_time = r_norm_elapse_time + n_norm_elapse_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NORM {norm_elapse_time} sec')

        # time record
        tagged_docs[0]['elapse_time'] = {
            'gnormplus_elapse_time': gnormplus_elapse_time,
            'biomedner_elapse_time':biomedner_elapse_time,
            'ner_elapse_time': ner_elapse_time,
            'r_norm_elapse_time':r_norm_elapse_time,
            'n_norm_elapse_time':n_norm_elapse_time,
            'norm_elapse_time':norm_elapse_time,
        } 

        # Delete temp files
        os.remove(input_gnormplus)
        os.remove(input_biomedner)
        os.remove(output_biomedner)
        # print(tagged_docs[0])
        return tagged_docs[0], maccrobat_entities

    # generate id for temporary files
    def generate_base_name(self, text):
        # add time.time() to avoid collision
        base_name = hashlib.sha224((text+str(time.time())).encode('utf-8')).hexdigest()
        return base_name

    async def async_ner(self, arguments):
        coroutines = [self._ner_wrap(*arg) for arg in arguments]
        result = await asyncio.gather(*coroutines)
        result = {k:v for e in result for k,v in e.items()} # merge
        return result

    async def _ner_wrap(self, ner_type, pubtator_file, output_biomedner, base_name, loop):
        if ner_type == 'gnormplus':
            # Run GNormPlus
            gnormplus_start_time = time.time()
            gnormplus_resp = await async_tell_inputfile(self.gnormplus_host,
                                            self.gnormplus_port,
                                            pubtator_file,
                                            loop)
            # Print time for GNormPlus
            gnormplus_elapse_time = time.time() - gnormplus_start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] GNormPlus {gnormplus_elapse_time} sec')

            return {"gnormplus_elapse_time": gnormplus_elapse_time,
                    "gnormplus_resp": gnormplus_resp}

        elif ner_type == 'biomedner':            
            # Run neural model
            start_time = time.time()
            biomedner_resp = await async_tell_inputfile(self.biomedner_host,
                                         self.biomedner_port,
                                         pubtator_file,
                                         loop)
            
            with open(output_biomedner, 'r', encoding='utf-8') as f:
                tagged_docs = [json.load(f)]

            num_entities = tagged_docs[0]['num_entities']
            if tagged_docs is None:
                return None

            assert len(tagged_docs) == 1
            biomedner_elapse_time = time.time() - start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] Multi-task NER {biomedner_elapse_time} sec, #entities: {num_entities}')

            return {"biomedner_elapse_time": biomedner_elapse_time,
                    "tagged_docs": tagged_docs,
                    "num_entities": num_entities}
            
        elif ner_type == 'maccrobat':
            # Run MacCrobat
            start_time = time.time()
            input_biomedner = os.path.join(self.biomedner_home, 'input',
                                f'{pubtator_file}.PubTator')
            pubtator_text = pubtator2dict_list(input_biomedner)[0]["abstract"]
            maccrobat_resp = await async_send_text_to_maccrobat_server(self.maccrobat_host,
                                            self.maccrobat_port,
                                            pubtator_text)
            
            print(maccrobat_resp)
            
            maccrobat_elapse_time = time.time() - start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] Maccrobat {maccrobat_elapse_time} sec')
            return {"maccrobat_elapse_time": maccrobat_elapse_time,
                    "maccrobat_resp": maccrobat_resp} 
            

async def async_send_text_to_maccrobat_server(host, port, text):
    reader, writer = await asyncio.open_connection(host, port)

    message = text.encode('utf-8')
    message_length = struct.pack('>H', len(message))
    writer.write(message_length + message)
    await writer.drain()

    response_length_data = await reader.readexactly(2)
    if len(response_length_data) < 2:
        print("Error: Server sent an incomplete response.")
        writer.close()
        await writer.wait_closed()
        return {}

    response_length = struct.unpack('>H', response_length_data)[0]
    response_data = await reader.readexactly(response_length)

    writer.close()
    await writer.wait_closed()
    return json.loads(response_data.decode('utf-8'))
                
async def async_tell_inputfile(host, port, inputfile, loop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        input_str = inputfile
        input_stream = struct.pack('>H', len(input_str)) + input_str.encode(
            'utf-8')
        sock.send(input_stream)
        # output_stream = sock.recv(512)
        output_stream = await loop.run_in_executor(None, sock.recv, 512) # for async
        resp = output_stream.decode('utf-8')[2:]

        sock.close()
        return resp
    except ConnectionRefusedError as e:
        print(e)
        return None
    except TimeoutError as e:
        print(e)
        return None
    except ConnectionResetError as e:
        print(e)
        return None

def sync_tell_inputfile(host, port, inputfile):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        input_str = inputfile
        input_stream = struct.pack('>H', len(input_str)) + input_str.encode(
            'utf-8')
        sock.send(input_stream)
        output_stream = sock.recv(512) # for sync
        # output_stream = await loop.run_in_executor(None, sock.recv, 512)
        resp = output_stream.decode('utf-8')[2:]

        sock.close()
        return resp
    except ConnectionRefusedError as e:
        print(e)
        return None
    except TimeoutError as e:
        print(e)
        return None
    except ConnectionResetError as e:
        print(e)
        return None

def delete_files(dirname):
    if not os.path.exists(dirname):
        return

    n_deleted = 0
    for f in os.listdir(dirname):
        f_path = os.path.join(dirname, f)
        if not os.path.isfile(f_path):
            continue
        # print('Delete', f_path)
        os.remove(f_path)
        n_deleted += 1
    print(dirname, n_deleted)
    
def get_synonyms(text, dictionary):
    synonyms = []
    for entry in dictionary:
        if entry['word'] == text:
            synonyms = entry['synonyms']
            break
    return synonyms
    
def transform_results(data):
    all_entities = []

    # Iterate through each annotation and extract required fields
    for annotation in data['annotations']:
        if annotation['prob'] > 0.5:
            entity = {
                'entity_group': annotation['obj'],  # Group by object type, e.g., 'Cell_type'
                'score': annotation['prob'],  # Probability score
                'word': annotation['mention'],  # Text mention
                'start': annotation['span']['begin'],  # Start position
                'end': annotation['span']['end'],  # End position
                'normalized_id': annotation['id'],  # Include the ID(s) associated with the entity
            }
            all_entities.append(entity)

    # Sort entities by start, end, and -score (negative score for descending order)
    all_entities.sort(key=lambda x: (x['start'], x['end'], -x['score']))

    # Filter out overlapping entities, keeping only the one with the highest score
    non_overlapping_entities = []
    last_end = -1

    for entity in all_entities:
        if entity['start'] >= last_end:
            non_overlapping_entities.append(entity)
            last_end = entity['end']

    return non_overlapping_entities

def get_synonyms_from_file(file_path, entity_ids):
    # Initialize an empty dictionary to store entities and their synonyms
    entity_dict = {}
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line at '||' to separate the IDs and the names
            parts = line.strip().split('||')
            if len(parts) > 1:
                # Split the first part by comma for multiple IDs
                ids = parts[0].split(',')
                # Split the second part by pipe to get all names/synonyms
                names = parts[1].split('|')
                
                # Populate the dictionary with each ID as a key and the names as values
                for nid in ids:
                    entity_dict[nid.strip()] = names
    
    # Fetch the synonyms using the entity ID provided
    synonyms = []
    for nid in entity_ids:
        synonyms.extend(entity_dict.get(nid.upper(), []))
    return synonyms

def get_gene_synonyms_from_file(file_path, entity_ids):
    # Initialize an empty dictionary to store gene entities and their synonyms
    gene_dict = {}
    

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line contains one ID and one synonym
            parts = line.strip().split('||')
            if len(parts) > 1:
                nid = parts[0].strip()
                synonym = parts[1].strip()
                if nid in gene_dict:
                    gene_dict[nid].append(synonym)
                else:
                    gene_dict[nid] = [synonym]
    
    # Fetch the synonyms using the entity IDs provided
    synonyms = []
    for nid in entity_ids:
        if nid in gene_dict:
            synonyms.extend(gene_dict[nid])
        else:
            synonyms.append("ID not found")
    return synonyms


def append_synonyms(ner_results, dict_paths):
    # Valid entity groups to consider for synonym addition
    valid_entity_groups = ['disease', 'gene', 'drug', 'species', 'cell line', 'cell type']
    
    # Entity group to dictionary file mapping
    dict_files = {
        'gene': dict_paths['gene'],
        'disease': dict_paths['disease'],
        'drug': dict_paths['drug'],  # Assuming 'drug' maps to 'ChemicalCompound'
        'species': dict_paths['species'],
        'cell_line': dict_paths['cell_line'],
        'cell_type': dict_paths['cell_type']
    }
    
    # Process each entity in the results
    for entity in ner_results:
        entity_group = entity['entity_group'].lower()
        if entity_group in valid_entity_groups:
            if entity_group in dict_files:
                file_path = dict_files[entity_group]
                if entity_group == 'gene':
                    entity['normalized_id'] = [nid.replace('NCBIGene:', 'EntrezGene:') for nid in entity['normalized_id']]
                    # Special processing for genes to convert NCBI to Entrez IDs
                    entity['synonyms'] = get_gene_synonyms_from_file(file_path, entity['normalized_id'])
                else:
                    # Generic processing for other entity types
                    entity['synonyms'] = get_synonyms_from_file(file_path, entity['normalized_id'])
            else:
                entity['synonyms'] = []
        else:
            entity['synonyms'] = []



if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_word_len', type=int, help='word max chars',
                           default=50)
    argparser.add_argument('--seed', type=int, help='seed value', default=2019)
    argparser.add_argument('--gnormplus_home',
                           help='GNormPlus home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'resources', 'GNormPlusJava'))
    argparser.add_argument('--gnormplus_host',
                           help='GNormPlus host', default='localhost')
    argparser.add_argument('--gnormplus_port', type=int,
                           help='GNormPlus port', default=18895)
    argparser.add_argument('--biomedner_home',
                           help='biomedical language model home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'resources', 'biomednerHome'))
    argparser.add_argument('--biomedner_host',
                           help='biomedical language model host', default='localhost')
    argparser.add_argument('--biomedner_port', type=int, 
                           help='biomedical language model port', default=18894)
    argparser.add_argument('--maccrobat_host', 
                           help='maccrobat host', default='localhost')
    argparser.add_argument('--maccrobat_port', type=int,
                            help='maccrobat port', default=18783)
    argparser.add_argument('--gene_norm_port', type=int,
                           help='GNormPlus port', default=18888)
    argparser.add_argument('--disease_norm_port', type=int,
                           help='Sieve port', default=18892)
    argparser.add_argument('--time_format',
                           help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
    argparser.add_argument("--use_neural_normalizer", action="store_true")
    argparser.add_argument("--keep_files", action="store_true")
    argparser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()

    biomedner = BioMedNER(
        max_word_len=args.max_word_len,
        seed=args.seed,
        gnormplus_home=args.gnormplus_home,
        gnormplus_host=args.gnormplus_host,
        gnormplus_port=args.gnormplus_port,
        gene_norm_port=args.gene_norm_port,
        disease_norm_port=args.disease_norm_port,
        biomedner_home=args.biomedner_home,
        biomedner_host=args.biomedner_host,
        biomedner_port=args.biomedner_port,
        maccrobat_host=args.maccrobat_host,
        maccrobat_port=args.maccrobat_port,
        time_format=args.time_format,
        use_neural_normalizer=args.use_neural_normalizer,
        keep_files=args.keep_files,
        no_cuda=args.no_cuda,
    )

    result = biomedner.annotate_text("Patients with microsatellite stable tumor and a tumor mutation burden (TMB) level measured at > 20 mutations per megabase pairs (MB)")
    print(result)