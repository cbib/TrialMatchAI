# server_script.py
import os
import json
import socket
import struct
import argparse
from datetime import datetime
from biomedner_init import BioMedNER
from ops import filter_entities, pubtator2dict_list

def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        for ent_type, entities in d['entities'].items():
            num_entities += len(entities)
    return num_entities

def biomedner_recognize(model, dict_path, base_name, biomedner_home, args):
    # Ensure input and output directories exist within biomedner_home
    input_dir = os.path.join(biomedner_home, 'input')
    output_dir = os.path.join(biomedner_home, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    input_mt_ner = os.path.join(biomedner_home, 'input',
                                f'{dict_path}.biomedner.PubTator')
    output_mt_ner = os.path.join(biomedner_home, 'output',
                                f'{dict_path}.biomedner.json')
    
    dict_list = pubtator2dict_list(input_mt_ner)

    res = model.recognize(
        input_dl=dict_list,
        base_name=base_name
    )

    if res is None:
        return None, 0

    num_filtered_species_per_doc = filter_entities(res)
    for n_f_spcs in num_filtered_species_per_doc:
        if n_f_spcs[1] > 0:
            print(datetime.now().strftime(args.time_format),
                  '[{}] Filtered {} species'.format(base_name, n_f_spcs[1]))
    num_entities = count_entities(res)

    res[0]['num_entities'] = num_entities
    # Write output str to a .PubTator format file
    with open(output_mt_ner, 'w', encoding='utf-8') as f:
        json.dump(res[0], f)

def run_server(model, args):
    host = args.biomedner_host
    port = args.biomedner_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(600)
        print(f"Server listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                message_length = struct.unpack('>H', conn.recv(2))[0]
                message = conn.recv(message_length).decode('utf-8')
                request = json.loads(message)
                biomedner_home = request["biomedner_home"]
                inputfile = request["inputfile"]
                base_name = inputfile.split('.')[0]
                base_name = base_name.replace("\x00A", "")
                
                biomedner_recognize(model, inputfile, base_name, biomedner_home, args)
                
                output_stream = struct.pack('>H', len(inputfile)) + inputfile.encode('utf-8')
                conn.send(output_stream)
                print(f"Response sent for {inputfile}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed for initialization', default=1)
    argparser.add_argument('--model_name_or_path')
    argparser.add_argument('--max_seq_length', type=int, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.', default=512)         
    argparser.add_argument('--biomedner_host', help='biomedical language model host', default='localhost')
    argparser.add_argument('--biomedner_port', type=int, help='biomedical language model port', default=18894)
    argparser.add_argument('--time_format', help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')    
    argparser.add_argument('--no_cuda', action="store_false", help="Avoid using CUDA when available")
    args = argparser.parse_args()
    mt_ner = BioMedNER(args)
    run_server(mt_ner, args)
