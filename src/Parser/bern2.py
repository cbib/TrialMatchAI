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


import pymongo
from pymongo import MongoClient

class RunBioMedNER():
    def __init__(self, 
        biomedner_home,
        biomedner_port,
        biomedner_host='localhost',
        time_format='[%d/%b/%Y %H:%M:%S.%f]',
        max_word_len=50, 
        seed=2019,
        keep_files=False,
        no_cuda=False):

        self.time_format = time_format

        print(datetime.now().strftime(self.time_format), 'BioMedNER LOADING..')
        random.seed(seed)
        np.random.seed(seed)

        if not os.path.exists('./output'):
            os.mkdir('output')

        # delete prev. version outputs
        if not keep_files:
            delete_files('./output')
            delete_files(os.path.join('./multi_ner', 'input'))
            delete_files(os.path.join('./multi_ner', 'tmp'))
            delete_files(os.path.join('./multi_ner', 'output'))

        self.biomedner_home = biomedner_home
        self.biomedner_host = biomedner_host
        self.biomedner_port = biomedner_port

        self.max_word_len = max_word_len

        print(datetime.now().strftime(self.time_format), 'BioMedNER LOADED..')
    
    def annotate_text(self, text, pmid=None):
        try:
            text = text.strip()
            base_name = self.generate_base_name(text) # for the name of temporary files
            text = self.preprocess_input(text, base_name)
            output = self.tag_entities(text, base_name)
            output['error_code'], output['error_message'] = 0, ""
        except Exception as e:
            errStr = traceback.format_exc()
            print(errStr)

            output = {"error_code": 1, "error_message": "Something went wrong. Try again."}

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

        input_biomedner = os.path.join(self.biomedner_home, 'input',
                                     f'{pubtator_file}.PubTator')
        output_biomedner = os.path.join(self.biomedner_home, 'output',
                                     f'{pubtator_file}.json')

        if not os.path.exists(self.biomedner_home + '/input'):
            os.mkdir(self.biomedner_home + '/input')
        if not os.path.exists(self.biomedner_home + '/output'):
            os.mkdir(self.biomedner_home + '/output')

        # Write input str to a .PubTator format file
        with open(input_biomedner, 'w', encoding='utf-8') as f:
            # only abstract
            f.write(f'{base_name}|t|\n')
            f.write(f'{base_name}|a|{text}\n\n')

        ner_start_time = time.time()
        
        arguments_for_coroutines = []
        loop = asyncio.new_event_loop()
        for ner_type in ['biomedner']:
            arguments_for_coroutines.append([ner_type, pubtator_file, output_biomedner, base_name, loop])
        async_result = loop.run_until_complete(self.async_ner(arguments_for_coroutines))
        loop.close()
        biomedner_elapse_time = async_result['biomedner_elapse_time']

        # get output result to merge
        tagged_docs = async_result['tagged_docs']
        num_entities = async_result['num_entities']
        
        ner_elapse_time = time.time() - ner_start_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NER {ner_elapse_time} sec')

        # time record
        tagged_docs[0]['elapse_time'] = {
            'biomedner_elapse_time':biomedner_elapse_time,
            'ner_elapse_time': ner_elapse_time,
        } 

        # Delete temp files
        os.remove(input_biomedner)
        os.remove(output_biomedner)
        
        return tagged_docs[0]

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
        if ner_type == 'biomedner':            
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

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_word_len', type=int, help='word max chars',
                           default=50)
    argparser.add_argument('--seed', type=int, help='seed value', default=2019)

    argparser.add_argument('--biomedner_home',
                           help='biomedical language model home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'biomedner', 'biomednerHome'))
    argparser.add_argument('--biomedner_host',
                           help='biomedical language model host', default='localhost')
    argparser.add_argument('--biomedner_port', type=int, 
                           help='biomedical language model port', default=18894)
    argparser.add_argument('--time_format',
                           help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
    argparser.add_argument("--keep_files", action="store_true")
    argparser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()
    biomedner = RunBioMedNER(
        max_word_len=args.max_word_len,
        seed=args.seed,
        biomedner_home=args.biomedner_home,
        biomedner_host=args.biomedner_host,
        biomedner_port=args.biomedner_port,
        time_format=args.time_format,
        keep_files=args.keep_files,
        no_cuda=args.no_cuda,
    )

    result = biomedner.annotate_text("monocytes")
    print(result)
