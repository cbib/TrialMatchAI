from typing import List
import math

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class BM25Retriever(BaseRetriever):
    documents: List[Document]
    k: int
    document_entity_index: dict
    idf_scores: dict
    average_doc_length: float

    def __init__(self, documents: List[Document], k: int, entity_extractor):
        super().__init__()
        self.documents = documents
        self.k = k
        self.entity_extractor = entity_extractor
        self._prepare_documents()

    def _prepare_documents(self):
        """Prepare documents by extracting entities and calculating necessary BM25 metrics."""
        num_docs = len(self.documents)
        doc_lengths = []
        df = {}
        
        # Extract entities and calculate document frequency (DF)
        for doc in self.documents:
            entities = self.entity_extractor.extract(doc.page_content)
            doc.entity_bag = entities
            doc_lengths.append(len(entities))
            
            unique_entities = set(entities)
            for entity in unique_entities:
                if entity in df:
                    df[entity] += 1
                else:
                    df[entity] = 1
        
        self.average_doc_length = sum(doc_lengths) / num_docs
        self.idf_scores = {term: math.log((num_docs - df[term] + 0.5) / (df[term] + 0.5)) for term in df}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Use BM25 to rank documents based on entities extracted from the query."""
        query_entities = self.entity_extractor.extract(query)
        scores = []

        for doc in self.documents:
            score = 0
            for entity in query_entities:
                if entity in doc.entity_bag:
                    term_frequency = doc.entity_bag.count(entity)
                    idf = self.idf_scores.get(entity, 0)
                    doc_length = len(doc.entity_bag)
                    score += idf * (term_frequency * (1.2 + 1) / (term_frequency + 1.2 * (1 - 0.75 + 0.75 * (doc_length / self.average_doc_length))))
            if score > 0:
                scores.append((score, doc))

        # Sort documents by their score and return top k
        sorted_docs = sorted(scores, key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs[:self.k]]

# Note: This implementation assumes the presence of an entity_extractor with an extract method.
