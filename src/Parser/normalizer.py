import os
import time
import socket
from concurrent.futures import ThreadPoolExecutor
from .normalizers.neural_normalizer import NeuralNormalizer
from .normalizers.normalizer_all import CellTypeNormalizer, ProcedureNormalizer, ChemicalNormalizer, SpeciesNormalizer, CellLineNormalizer, SignSymptomNormalizer

time_format = '[%d/%b/%Y %H:%M:%S.%f]'

class Normalizer:
    def __init__(self, use_neural_normalizer, gene_port=18888, disease_port=18892, no_cuda=False):
        self.BASE_DIR = 'src/Parser/resources/normalization/'
        self.NORM_INPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'inputs/disease'),
            'gene': os.path.join(self.BASE_DIR, 'inputs/gene'),
        }
        self.NORM_OUTPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'outputs/disease'), 
            'gene': os.path.join(self.BASE_DIR, 'outputs/gene'),
        }

        self.NORM_DICT_PATH = {
            'drug': os.path.join(self.BASE_DIR, 'dictionary/dict_ChemicalsDrugs.txt'),
            'gene': 'setup.txt',
            'species': os.path.join(self.BASE_DIR, 'dictionary/dict_Species.txt'),
            'cell line': os.path.join(self.BASE_DIR, 'dictionary/dict_CellLine.txt'),
            'cell type': os.path.join(self.BASE_DIR, 'dictionary/dict_CellType.txt'),
            'procedure': os.path.join(self.BASE_DIR, 'dictionary/dict_Procedures.txt'),  
            'sign symptom': os.path.join(self.BASE_DIR, 'dictionary/dict_SignSymptom.txt'),
        }

        self.NEURAL_NORM_MODEL_PATH = {
            'disease': 'dmis-lab/biosyn-sapbert-bc5cdr-disease',
            'sign symptom': 'dmis-lab/biosyn-sapbert-bc5cdr-disease',
            'drug': 'dmis-lab/biosyn-sapbert-bc5cdr-chemical',
            'gene': 'dmis-lab/biosyn-sapbert-bc2gn',
        }
        self.NEURAL_NORM_CACHE_PATH = {
            'disease': os.path.join(self.BASE_DIR, 'normalizers/neural_norm_caches/dict_Disease_20210630.txt.pk'),
            'sign symptom': os.path.join(self.BASE_DIR, 'normalizers/neural_norm_caches/dict_Disease_20210630.txt.pk'),
            'drug': os.path.join(self.BASE_DIR, 'normalizers/neural_norm_caches/dict_ChemicalCompound_20210630.txt.pk'),
            'gene': os.path.join(self.BASE_DIR, 'normalizers/neural_norm_caches/dict_Gene.txt.pk')
        }

        self.NORM_MODEL_VERSION = 'N/A'
        self.HOST = '127.0.0.1'
        self.GENE_PORT = gene_port
        self.DISEASE_PORT = disease_port
        self.NO_ENTITY_ID = 'CUI-less'
        self.use_neural_normalizer = use_neural_normalizer

        self.chemical_normalizer = None
        self.species_normalizer = None
        self.cellline_normalizer = None
        self.celltype_normalizer = None
        self.procedure_normalizer = None
        self.sign_symptom_normalizer = None

        self.neural_disease_normalizer = None
        self.neural_chemical_normalizer = None
        self.neural_gene_normalizer = None

        # Load normalizers concurrently
        with ThreadPoolExecutor() as executor:
            future_dict = {
                'chemical': executor.submit(ChemicalNormalizer, self.NORM_DICT_PATH['drug']),
                'species': executor.submit(SpeciesNormalizer, self.NORM_DICT_PATH['species']),
                'cellline': executor.submit(CellLineNormalizer, self.NORM_DICT_PATH['cell line']),
                'celltype': executor.submit(CellTypeNormalizer, self.NORM_DICT_PATH['cell type']),
                'procedure': executor.submit(ProcedureNormalizer, self.NORM_DICT_PATH['procedure']),
                'sign symptom': executor.submit(SignSymptomNormalizer, self.NORM_DICT_PATH['sign symptom']),
            }

            self.chemical_normalizer = future_dict['chemical'].result()
            self.species_normalizer = future_dict['species'].result()
            self.cellline_normalizer = future_dict['cellline'].result()
            self.celltype_normalizer = future_dict['celltype'].result()
            self.procedure_normalizer = future_dict['procedure'].result()
            self.sign_symptom_normalizer = future_dict['sign symptom'].result()

        if self.use_neural_normalizer:
            with ThreadPoolExecutor() as executor:
                future_dict = {
                    'neural_disease': executor.submit(NeuralNormalizer, model_name_or_path=self.NEURAL_NORM_MODEL_PATH['disease'], cache_path=self.NEURAL_NORM_CACHE_PATH['disease'], no_cuda=no_cuda),
                    'neural_sign symptom': executor.submit(NeuralNormalizer, model_name_or_path=self.NEURAL_NORM_MODEL_PATH['sign symptom'], cache_path=self.NEURAL_NORM_CACHE_PATH['sign symptom'], no_cuda=no_cuda),
                    'neural_chemical': executor.submit(NeuralNormalizer, model_name_or_path=self.NEURAL_NORM_MODEL_PATH['drug'], cache_path=self.NEURAL_NORM_CACHE_PATH['drug'], no_cuda=no_cuda),
                    'neural_gene': executor.submit(NeuralNormalizer, model_name_or_path=self.NEURAL_NORM_MODEL_PATH['gene'], cache_path=self.NEURAL_NORM_CACHE_PATH['gene'], no_cuda=no_cuda),
                }
                self.neural_disease_normalizer = future_dict['neural_disease'].result()
                self.neural_sign_symptom_normalizer = future_dict['neural_sign symptom'].result()
                self.neural_chemical_normalizer = future_dict['neural_chemical'].result()
                self.neural_gene_normalizer = future_dict['neural_gene'].result()

    def normalize(self, base_name, doc_dict_list):
        start_time = time.time()

        names = dict()
        saved_items = list()
        ent_cnt = 0
        abs_cnt = 0

        for item in doc_dict_list:
            content = item['abstract']
            entities = item['entities']

            abs_cnt += 1

            for ent_type, locs in entities.items():
                ent_cnt += len(locs)
                for loc in locs:
                    loc['end'] += 1
                    name = content[loc['start']:loc['end']]
                    if ent_type in names:
                        names[ent_type].append([name, len(saved_items)])
                    else:
                        names[ent_type] = [[name, len(saved_items)]]

            item['norm_model'] = self.NORM_MODEL_VERSION
            saved_items.append(item)

        results = list()
        with ThreadPoolExecutor() as executor:
            futures = []
            for ent_type in names.keys():
                futures.append(executor.submit(self.run_normalizers_wrap, ent_type, base_name, names, saved_items, results))
            for future in futures:
                future.result()

        for ent_type, type_oids in results:
            oid_cnt = 0
            for saved_item in saved_items:
                for loc in saved_item['entities'][ent_type]:
                    loc['id'] = type_oids[oid_cnt]
                    loc['is_neural_normalized'] = False
                    oid_cnt += 1

        return saved_items

    def neural_normalize(self, ent_type, tagged_docs):
        abstract = tagged_docs[0]['abstract']
        entities = tagged_docs[0]['entities'][ent_type]
        entity_names = [abstract[e['start']:e['end']] for e in entities]
        cuiless_entity_names = []
        for entity, entity_name in zip(entities, entity_names):
            if entity['id'] == self.NO_ENTITY_ID:
                cuiless_entity_names.append(entity_name)
        cuiless_entity_names = list(set(cuiless_entity_names))
        
        if len(cuiless_entity_names) == 0:
            return tagged_docs

        if ent_type == 'disease':
            norm_entities = self.neural_disease_normalizer.normalize(names=cuiless_entity_names)
        elif ent_type == 'drug':
            norm_entities = self.neural_chemical_normalizer.normalize(names=cuiless_entity_names)
        elif ent_type == 'gene':
            norm_entities = self.neural_gene_normalizer.normalize(names=cuiless_entity_names)
        elif ent_type == 'sign symptom':
            norm_entities = self.neural_sign_symptom_normalizer.normalize(names=cuiless_entity_names)
        
        cuiless_entity2norm_entities = {c:n for c, n in zip(cuiless_entity_names,norm_entities)}
        for entity, entity_name in zip(entities, entity_names):
            if entity_name in cuiless_entity2norm_entities:
                cui = cuiless_entity2norm_entities[entity_name][0]
                entity['id'] = cui if cui != -1 else self.NO_ENTITY_ID
                entity['is_neural_normalized'] = True
            else:
                entity['is_neural_normalized'] = False
        
        return tagged_docs

    def run_normalizers_wrap(self, ent_type, base_name, names, saved_items, results):
        results.append((ent_type, self.run_normalizer(ent_type, base_name, names, saved_items)))

    def run_normalizer(self, ent_type, base_name, names, saved_items):
        start_time = time.time()
        name_ptr = names[ent_type]
        oids = list()
        bufsize = 4

        base_thread_name = base_name
        input_filename = base_thread_name + '.concept'
        output_filename = base_thread_name + '.oid'

        if ent_type in ['disease']:
                norm_inp_path = os.path.join(self.NORM_INPUT_DIR[ent_type], input_filename)
                norm_abs_path = os.path.join(self.NORM_INPUT_DIR[ent_type], base_thread_name + '.txt')
                with open(norm_inp_path, 'w') as norm_inp_f:
                    for name, _ in name_ptr:
                        norm_inp_f.write(name + '\n')
                with open(norm_abs_path, 'w') as _:
                    pass

                s = socket.socket()
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    s.connect((self.HOST, self.DISEASE_PORT))
                    s.send('{}'.format(base_thread_name).encode('utf-8'))
                    s.recv(bufsize)
                except ConnectionRefusedError as cre:
                    os.remove(norm_inp_path)
                    os.remove(norm_abs_path)
                    s.close()
                    return oids
                s.close()

                norm_out_path = os.path.join(self.NORM_OUTPUT_DIR[ent_type], output_filename)
                if os.path.exists(norm_out_path):
                    with open(norm_out_path, 'r') as norm_out_f:
                        for line in norm_out_f:
                            oid = line[:-1]
                            oids.append(oid if oid != self.NO_ENTITY_ID else self.NO_ENTITY_ID)
                    os.remove(norm_out_path)
                else:
                    for _ in range(len(name_ptr)):
                        oids.append(self.NO_ENTITY_ID)

        elif ent_type in ['drug']:
            names = [ptr[0] for ptr in name_ptr]
            preds = self.chemical_normalizer.normalize(names)
            oids.extend(preds)

        elif ent_type == 'mutation':
            pass

        elif ent_type == 'species':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.species_normalizer.normalize(names)
            for pred in preds:
                if pred != self.NO_ENTITY_ID:
                    pred = int(pred) // 100
                    oids.append('NCBI:txid{}'.format(pred))
                else:
                    oids.append(self.NO_ENTITY_ID)
        
        elif ent_type == 'cell line':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.cellline_normalizer.normalize(names)
            oids.extend(preds)

        elif ent_type == 'cell type':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.celltype_normalizer.normalize(names)
            oids.extend(preds)

        elif ent_type == 'gene':
            s = socket.socket()
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.connect((self.HOST, self.GENE_PORT))
            except ConnectionRefusedError as cre:
                s.close()
                return oids

            norm_inp_path = os.path.join(self.NORM_INPUT_DIR[ent_type], input_filename)
            norm_abs_path = os.path.join(self.NORM_INPUT_DIR[ent_type], base_thread_name + '.txt')

            space_type = ' ' + ent_type
            with open(norm_inp_path, 'w') as norm_inp_f, open(norm_abs_path, 'w') as norm_abs_f:
                for saved_item in saved_items:
                    entities = saved_item['entities'][ent_type]
                    if len(entities) == 0:
                        continue

                    abstract_title = saved_item['abstract']
                    ent_names = [abstract_title[loc['start']:loc['end']] for loc in entities]

                    norm_abs_f.write(saved_item['pmid'] + '||' + abstract_title + '\n')
                    norm_inp_f.write('||'.join(ent_names) + '\n')

            gene_input_dir = os.path.abspath(self.NORM_INPUT_DIR[ent_type])
            gene_output_dir = os.path.abspath(self.NORM_OUTPUT_DIR[ent_type])
            setup_dir = self.NORM_DICT_PATH[ent_type]

            jar_args = '\t'.join([gene_input_dir, gene_output_dir, setup_dir, '9606', base_thread_name]) + '\n'
            s.send(jar_args.encode('utf-8'))
            s.recv(bufsize)
            s.close()

            norm_out_path = os.path.join(gene_output_dir, output_filename)
            if os.path.exists(norm_out_path):
                with open(norm_out_path, 'r') as norm_out_f, open(norm_inp_path, 'r') as norm_in_f:
                    for line, input_l in zip(norm_out_f, norm_in_f):
                        gene_ids, gene_mentions = line[:-1].split('||'), input_l[:-1].split('||')
                        for gene_id, gene_mention in zip(gene_ids, gene_mentions):
                            eid = "EntrezGene:" + gene_id if gene_id.lower() != 'cui-less' else self.NO_ENTITY_ID
                            oids.append(eid)

                os.remove(norm_out_path)
            else:
                for _ in range(len(name_ptr)):
                    oids.append(self.NO_ENTITY_ID)

            os.remove(norm_inp_path)
            os.remove(norm_abs_path)
        
        elif ent_type in ['diagnostic test', 'treatment', 'radiology', 'surgical procedure', 'laboratory test', 'genomic analysis technique']:
            names = [ptr[0] for ptr in name_ptr]
            preds = self.procedure_normalizer.normalize(names)
            oids.extend(preds)
            
        elif ent_type == 'sign symptom':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.sign_symptom_normalizer.normalize(names)
            oids.extend(preds)

        else:
            names = [ptr[0] for ptr in name_ptr]
            oids.extend([self.NO_ENTITY_ID] * len(names))

        assert len(oids) == len(name_ptr), '{} vs {} in {}'.format(len(oids), len(name_ptr), ent_type)

        return oids
