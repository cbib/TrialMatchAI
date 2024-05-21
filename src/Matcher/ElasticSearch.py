import os
import json
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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
        self.existing_nct_ids = self.get_existing_nct_ids()

    def get_existing_nct_ids(self):
        try:
            query = {
                "_source": ["nct_id"],
                "query": {"match_all": {}},
                "size": 1000  # Fetch 1000 documents per batch
            }

            nct_ids = set()
            response = self.es_client.search(
                index=self.index_name,
                body=query,
                scroll='2m'  # Keep the search context for 2 minutes
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while len(hits) > 0:
                for hit in hits:
                    nct_ids.add(hit['_source']['nct_id'])

                response = self.es_client.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            # Clear the scroll context
            self.es_client.clear_scroll(scroll_id=scroll_id)

            return nct_ids

        except NotFoundError:
            return set()

    def index_data(self, documents, refresh=True):
        if not self.es_client.indices.exists(index=self.index_name):
            self.create_index()
        else:
            print(f"Index {self.index_name} already exists. Skipping index creation.")

        requests = []
        for doc in documents:
            if doc['nct_id'] in self.existing_nct_ids:
                continue
            vector = self.embedder.get_embeddings(doc['criterion'])
            requests.append({
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc['nct_id'],
                "criterion": doc['criterion'],
                "criterion_vector": vector,
                "entities": doc['entities'],
                "eligibility_type": doc['eligibility_type'],
                "nct_id": doc['nct_id']
            })

        try:
            success, failed = bulk(self.es_client, requests, raise_on_error=False)
            print(f"Successfully indexed {len(success)} documents.")
            if failed:
                print("Some documents failed to index:", failed)
        except Exception as e:
            print("An error occurred during indexing:", str(e))

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
                    "criterion": {"type": "text", "analyzer": "standard_lowercase"},
                    "criterion_vector": {"type": "dense_vector", "dims": vector_dims},
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
    es_url = "https://4616afc5fbda42dfa82407bdaf369e18.us-central1.gcp.cloud.es.io"
    username = "elastic"
    password = "gWQzXpyXFfQp9q3tWgBwCGNG"

    es_client = Elasticsearch(
        hosts=[es_url],
        http_auth=(username, password)
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
    indexer.index_data(documents)
