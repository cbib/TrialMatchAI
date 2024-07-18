
import spacy
import re
import os
from datetime import datetime

class CellTypeNormalizer(object):
    def __init__(self, dict_path, nlp):
        self.NO_ENTITY_ID = 'CUI-less'
        self.nlp = nlp

        # Create dictionary for exact match
        self.ct2oid = dict()

        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Loading cell type dictionary...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                oid, names = line.strip().split('||')
                names = names.split('|')
                for name in names:
                    self.ct2oid[name.lower()] = oid
        
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Cell type dictionary loaded.')

    def normalize(self, names):
        oids = list()
        for name in names:
            normalized_name = self.get_tmchem_name(name)
            if normalized_name in self.ct2oid:
                oids.append(self.ct2oid[normalized_name])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids
    
    def get_tmchem_name(self, name):
        # 1. lowercase, 2. removes all whitespace and punctuation, 3. lemmatize
        cleaned_name = re.sub(r'[^\w\s-]', '', name.lower()).replace(' ', '')
        doc = self.nlp(cleaned_name)
        lemmatized_name = ''.join([token.lemma_ for token in doc])
        return lemmatized_name

class ProcedureNormalizer(object):
    def __init__(self, dict_path, nlp):
        self.NO_ENTITY_ID = 'CUI-less'
        self.nlp = nlp

        # Create dictionary for exact match
        self.procedure2oid = dict()

        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Loading procedure dictionary...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                oid, names = line.strip().split('||')
                names = names.split('|')
                for name in names:
                    self.procedure2oid[name.lower()] = oid
        
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Procedure dictionary loaded.')

    def normalize(self, names):
        oids = list()
        for name in names:
            normalized_name = self.get_tmchem_name(name)
            if normalized_name in self.procedure2oid:
                oids.append(self.procedure2oid[normalized_name])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids
    
    def get_tmchem_name(self, name):
        # 1. lowercase, 2. removes all whitespace and punctuation, 3. lemmatize
        cleaned_name = re.sub(r'[^\w\s-]', '', name.lower()).replace(' ', '')
        doc = self.nlp(cleaned_name)
        lemmatized_name = ''.join([token.lemma_ for token in doc])
        return lemmatized_name

class ChemicalNormalizer(object):
    def __init__(self, dict_path, nlp):
        self.NO_ENTITY_ID = 'CUI-less'
        self.nlp = nlp

        # Create dictionary for exact match
        self.chem2oid = dict()

        # Log the size of the dictionary file
        dict_size = os.path.getsize(dict_path)
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), f'Chemical dictionary file size: {dict_size} bytes')

        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Loading chemical dictionary...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                oid, names = line.strip().split('||')
                names = names.split('|')
                for name in names:
                    self.chem2oid[name.lower()] = oid
        
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Chemical dictionary loaded.')

    def normalize(self, names):
        oids = list()
        for name in names:
            normalized_name = self.get_tmchem_name(name)
            if normalized_name in self.chem2oid:
                oids.append(self.chem2oid[normalized_name])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids
    
    def get_tmchem_name(self, name):
        # 1. lowercase, 2. removes all whitespace and punctuation, 3. lemmatize
        cleaned_name = re.sub(r'[^\w\s-]', '', name.lower()).replace(' ', '')
        doc = self.nlp(cleaned_name)
        lemmatized_name = ''.join([token.lemma_ for token in doc])
        return lemmatized_name

    
class CellLineNormalizer(object):
    def __init__(self, dict_path):
        self.NO_ENTITY_ID = 'CUI-less'

        # Create dictionary for exact match
        self.cl2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    self.cl2oid[name] = oid

    def normalize(self, names):
        oids = list()
        for name in names:
            if name in self.cl2oid:
                oids.append(self.cl2oid[name])
            elif name.lower() in self.cl2oid:
                oids.append(self.cl2oid[name.lower()])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids
    
class SpeciesNormalizer(object):
    def __init__(self, dict_path):
        self.NO_ENTITY_ID = 'CUI-less'

        # Create dictionary for exact match
        self.species2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    # a part of tmChem normalization
                    self.species2oid[name] = oid

    def normalize(self, names):
        oids = list()
        for name in names:
            if name in self.species2oid:
                oids.append(self.species2oid[name])
            elif name.lower() in self.species2oid:
                oids.append(self.species2oid[name.lower()])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids