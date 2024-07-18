import spacy
import re

# Load only the lemmatizer component
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("lemmatizer")
nlp.initialize()

class CellTypeNormalizer(object):
    def __init__(self, dict_path):
        self.NO_ENTITY_ID = 'CUI-less'

        # Create dictionary for exact match
        self.ct2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line.strip().split('||')
                names = names.split('|')
                for name in names:
                    normalized_name = self.get_tmchem_name(name)
                    self.ct2oid[normalized_name] = oid

    def normalize(self, names):
        oids = []
        for name in names:
            normalized_name = self.get_tmchem_name(name)
            if normalized_name in self.ct2oid:
                oids.append(self.ct2oid[normalized_name])
            else:
                oids.append(self.NO_ENTITY_ID)
        return oids
    
    def get_tmchem_name(self, name):
        # Lowercase and remove all whitespace and punctuation
        cleaned_name = re.sub(r'[^\w\s-]', '', name.lower()).replace(' ', '')
        # Use SpaCy to lemmatize the cleaned name
        doc = nlp(cleaned_name)
        lemmatized_name = ''.join([token.lemma_ for token in doc])
        return lemmatized_name