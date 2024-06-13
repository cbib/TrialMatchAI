import os
import json
import hashlib
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class SentenceEmbedder:
    def __init__(self, model_name: str = 'dmlls/all-mpnet-base-v2-negation'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentence: str) -> list:
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

class DocumentIndexer:
    def __init__(self, es_client, index_name, embedder, vector_dims):
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.vector_dims = vector_dims
        self.existing_criteria_ids = self.get_existing_criteria_ids()

    def generate_criteria_id(self, nct_id, criterion):
        return hashlib.sha256(f"{nct_id}_{criterion}".encode('utf-8')).hexdigest()

    def get_existing_criteria_ids(self):
        try:
            query = {
                "_source": ["criteria_id"],
                "query": {"match_all": {}},
                "size": 1000  # Fetch 1000 documents per batch
            }

            criteria_ids = set()
            response = self.es_client.search(
                index=self.index_name,
                body=query,
                scroll='5m'  # Keep the search context for 3 minutes
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while len(hits) > 0:
                for hit in hits:
                    criteria_ids.add(hit['_source']['criteria_id'])

                response = self.es_client.scroll(
                    scroll_id=scroll_id,
                    scroll='3m'
                )
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            # Clear the scroll context
            self.es_client.clear_scroll(scroll_id=scroll_id)

            return criteria_ids

        except NotFoundError:
            return set()

    def index_data(self, documents, refresh=True, batch_size=100, max_workers=4):
        if not self.es_client.indices.exists(index=self.index_name):
            self.create_index()
        else:
            print(f"Index {self.index_name} already exists. Skipping index creation.")

        def index_batch(batch):
            requests = []
            for doc in batch:
                criteria_id = self.generate_criteria_id(doc['nct_id'], doc['criterion'])
                if criteria_id not in self.existing_criteria_ids:
                    vector = self.embedder.get_embeddings(doc['criterion'])
                    requests.append({
                        "_op_type": "index",
                        "_index": self.index_name,
                        "criteria_id": criteria_id,
                        "criterion": doc['criterion'],
                        "criterion_vector": vector,
                        "entities": doc['entities'],
                        "eligibility_type": doc['eligibility_type'],
                        "nct_id": doc['nct_id']
                    })
            if requests:
                return bulk(self.es_client, requests, raise_on_error=False)
            return 0, []

        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(index_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    success_count, failed = future.result()
                    results.append((success_count, failed))
                except Exception as e:
                    print(f"An error occurred during indexing batch: {str(e)}")
        
        total_success = sum(success_count for success_count, _ in results)
        total_failed = sum(len(failed) for _, failed in results)
        print(f"Successfully indexed {total_success} documents.")
        if total_failed > 0:
            print(f"{total_failed} documents failed to index.")
        
        if refresh:
            self.es_client.indices.refresh(index=self.index_name)

    def create_index(self):
        create_index(self.es_client, self.index_name, self.vector_dims)

    @staticmethod
    def prepare_documents(folder_path):
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    nct_id = data['nct_id']
                    for criterion in data['criteria']:
                        document = {
                            'criterion': criterion['criterion'],
                            'entities': criterion['entities'],
                            'eligibility_type': criterion['type'],
                            'nct_id': nct_id
                        }
                        documents.append(document)
        return documents

def create_index(es_client: Elasticsearch, index_name: str, vector_dims: int):
    es_client.indices.create(
        index=index_name,
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "standard_lowercase": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "criteria_id": {"type": "keyword"},
                    "criterion": {"type": "text", "analyzer": "standard_lowercase"},
                    "criterion_vector": {
                        "type": "dense_vector",
                        "dims": vector_dims,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100
                        }
                    },
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "normalized_id": {"type": "keyword"},
                            "synonyms": {"type": "text", "analyzer": "standard_lowercase"},
                            "is_negated": {"type": "keyword"},
                            "entity": {"type": "text", "analyzer": "standard_lowercase"},
                            "class": {"type": "keyword"}
                        }
                    },
                    "nct_id": {"type": "keyword", "index": False}
                }
            }
        }
    )

if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="certs/ca/ca.crt",
        basic_auth=(username, password),
        max_retries=50,
        request_timeout=60,
        retry_on_timeout=True
    )

    info = es_client.info()
    print(info)

    embedder = SentenceEmbedder() 
    vector_dims = 768 

    folder_path = '../../data/ner_trial/'
    documents = DocumentIndexer.prepare_documents(folder_path)
    print(f"Prepared {len(documents)} documents for indexing.")

    index_name = "eligibility_criteria"
    indexer = DocumentIndexer(es_client, index_name, embedder, vector_dims)
    indexer.index_data(documents, batch_size=100, max_workers=4)
