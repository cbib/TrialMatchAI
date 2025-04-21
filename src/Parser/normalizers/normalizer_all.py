import re
import os
from datetime import datetime
from rapidfuzz import fuzz

class BaseNormalizer:
    """
    Base class for flexible normalization with exact and similarity-based matching.
    """
    def __init__(self, dict_path, nlp=None, similarity_threshold=0.5):
        self.NO_ENTITY_ID = 'CUI-less'
        self.similarity_threshold = similarity_threshold
        self.nlp = nlp
        self.entity2oid = dict()

        self.load_dictionary(dict_path)

    def load_dictionary(self, dict_path):
        """
        Load dictionary from the given path and normalize keys for matching.
        """
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Loading dictionary...')
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line.strip().split('||')
                names = names.split('|')
                for name in names:
                    normalized_name = self.normalize_string(name)
                    self.entity2oid[normalized_name] = oid
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), 'Dictionary loaded.')

    def normalize(self, names):
        """
        Normalize a list of names and return their corresponding IDs.
        """
        oids = list()
        for name in names:
            normalized_name = self.get_tmchem_name(name)

            # Exact match
            if normalized_name in self.entity2oid:
                oids.append(self.entity2oid[normalized_name])
            else:
                # Flexible match based on similarity score
                best_match = self.find_best_match(normalized_name)
                if best_match:
                    oids.append(self.entity2oid[best_match])
                else:
                    oids.append(self.NO_ENTITY_ID)

        return oids

    def get_tmchem_name(self, name):
        """
        Normalize a name using lowercasing, punctuation removal, and lemmatization (if NLP is enabled).
        """
        cleaned_name = re.sub(r'[^\w\s-]', '', name.lower())  # Remove punctuation but keep hyphens
        if self.nlp:
            doc = self.nlp(cleaned_name)
            lemmatized_name = ' '.join([token.lemma_ for token in doc])
        else:
            lemmatized_name = cleaned_name
        return self.normalize_string(lemmatized_name)

    def normalize_string(self, text):
        """
        Normalize a string for matching by removing spaces and hyphens.
        """
        return re.sub(r'[\s-]', '', text.lower())

    def find_best_match(self, normalized_name):
        """
        Find the best match for a given normalized name based on a similarity score.
        """
        best_match = None
        highest_score = 0

        for key in self.entity2oid.keys():
            score = self.similarity_score(normalized_name, key)
            if score > self.similarity_threshold and score > highest_score:
                best_match = key
                highest_score = score

        return best_match

    def similarity_score(self, name1, name2):
            """
            Use RapidFuzz's token_sort_ratio for fast and accurate similarity scoring.
            """
            return fuzz.token_sort_ratio(name1, name2) / 100  # Normalize score to 0-1 range





class CellTypeNormalizer(BaseNormalizer):
    pass


class ProcedureNormalizer(BaseNormalizer):
    pass


class ChemicalNormalizer(BaseNormalizer):
    def load_dictionary(self, dict_path):
        """
        Specialized dictionary loader for chemical normalizer with file size logging.
        """
        dict_size = os.path.getsize(dict_path)
        print(datetime.now().strftime('[%d/%b/%Y %H:%M:%S.%f]'), f'Chemical dictionary file size: {dict_size} bytes')
        super().load_dictionary(dict_path)


class CellLineNormalizer(BaseNormalizer):
    def get_tmchem_name(self, name):
        """
        Simplified name normalization for cell line data.
        """
        return self.normalize_string(name)


class SpeciesNormalizer(BaseNormalizer):
    def get_tmchem_name(self, name):
        """
        Simplified name normalization for species data.
        """
        return self.normalize_string(name)
    
class SignSymptomNormalizer(BaseNormalizer):
    pass

