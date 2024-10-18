import os
import json
import time
import re
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import dateutil
from dateutil import parser

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class SentenceEmbedder:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text: str) -> list:
        if not text:
            return None  # Return None for empty or None text fields
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding.squeeze().tolist()

    def preprocess_text(self, text: str) -> str:
        if not text:
            return text
        # Remove extra white spaces, tabs, and carriage returns
        text = re.sub(r'\s+', ' ', text)
        # Strip leading and trailing spaces
        text = text.strip()
        return text
    
    def convert_date_to_iso(self, date_str: str) -> str:
        """Converts a date string to ISO 8601 format (YYYY-MM-DD)."""
        if not date_str:  # Handle None or empty date strings
            return None

        try:
            # Attempt to parse the date string
            parsed_date = dateutil.parser.parse(date_str)
            return parsed_date.date().isoformat()  # Return in YYYY-MM-DD format
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error parsing date '{date_str}': {str(e)}")
            return None

    def convert_age_to_years(self, age_str: str) -> str:
        """Converts various age formats to a standard format in years."""
        if not age_str:  # Check if age_str is None or an empty string
            return None  # Return None for missing or empty age strings

        age_str = age_str.lower().strip()  # Normalize the input string
        
        try:
            if 'year' in age_str:
                years = float(re.search(r'\d+', age_str).group())
            elif 'month' in age_str:
                months = float(re.search(r'\d+', age_str).group())
                years = months / 12
            elif 'week' in age_str:
                weeks = float(re.search(r'\d+', age_str).group())
                years = weeks / 52
            elif 'day' in age_str:
                days = float(re.search(r'\d+', age_str).group())
                years = days / 365
            else:
                return None  # Return None for unknown or unhandled formats

            # Return the age as a string with " years"
            return f"{years:.2f}"
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error parsing age '{age_str}': {str(e)}")
            return None  # Return None if there's an error in parsing the age


class DocumentIndexer:
    def __init__(self, es_client, index_name, embedder, vector_dims):
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.vector_dims = vector_dims
        self.existing_document_ids = self.get_existing_document_ids()

    def get_existing_document_ids(self):
        try:
            query = {
                "_source": ["nct_id"],
                "query": {"match_all": {}},
                "size": 1000
            }

            document_ids = set()
            response = self.es_client.search(
                index=self.index_name,
                body=query,
                scroll='5m'
            )

            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

            while len(hits) > 0:
                for hit in hits:
                    document_ids.add(hit['_source']['nct_id'])

                response = self.es_client.scroll(
                    scroll_id=scroll_id,
                    scroll='3m'
                )
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']

            self.es_client.clear_scroll(scroll_id=scroll_id)
            return document_ids

        except NotFoundError:
            return set()

    def embed_and_prepare_document(self, doc):
        try:
            # Preprocess and embed text fields
            if 'condition' in doc:
                doc['condition'] = self.embedder.preprocess_text(doc['condition'])
                doc['condition_vector'] = self.embedder.get_embeddings(doc['condition'])
                
            # Convert date fields to ISO 8601 format
            if 'start_date' in doc:
                doc['start_date'] = self.embedder.convert_date_to_iso(doc['start_date'])
            if 'completion_date' in doc:
                doc['completion_date'] = self.embedder.convert_date_to_iso(doc['completion_date'])

            # Convert age fields to years
            if 'minimum_age' in doc:
                doc['minimum_age'] = self.embedder.convert_age_to_years(doc['minimum_age'])
            if 'maximum_age' in doc:
                doc['maximum_age'] = self.embedder.convert_age_to_years(doc['maximum_age'])

            return doc
        except Exception as e:
            print(f"Error processing document {doc.get('nct_id', 'unknown')}: {str(e)}")
            return None

    def index_data(self, documents, refresh=True, batch_size=100, max_workers=8):
        if not self.es_client.indices.exists(index=self.index_name):
            self.create_index()
        else:
            print(f"Index {self.index_name} already exists. Skipping index creation.")

        def index_batch(batch):
            requests = []
            for doc in batch:
                if doc and doc['nct_id'] not in self.existing_document_ids:
                    requests.append({
                        "_op_type": "index",
                        "_index": self.index_name,
                        "_id": doc['nct_id'],  # Use nct_id as the document ID
                        "_source": doc
                    })
                    self.existing_document_ids.add(doc['nct_id'])

            if requests:
                try:
                    print(f"Indexing batch of size {len(requests)}")
                    success, failed = bulk(self.es_client, requests, raise_on_error=False)
                    if failed:
                        print(f"Failed to index {len(failed)} documents. Errors: {failed}")
                    return success, failed
                except Exception as e:
                    print(f"An error occurred during bulk indexing: {str(e)}")
                    return 0, []
            return 0, []

        # Process documents in parallel, embedding and indexing each batch
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"Total batches created: {len(batches)}")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in batches:
                # Process embeddings in parallel
                futures = [executor.submit(self.embed_and_prepare_document, doc) for doc in batch]
                completed_batch = [future.result() for future in as_completed(futures)]
                # Filter out any documents that failed during embedding
                completed_batch = [doc for doc in completed_batch if doc]

                # Index the completed batch
                success_count, failed = index_batch(completed_batch)
                print(f"Indexed {success_count} documents in this batch.")
                results.append((success_count, failed))

        total_success = sum(success_count for success_count, _ in results)
        total_failed = sum(len(failed) for _, failed in results)
        print(f"Successfully indexed {total_success} documents.")
        if total_failed > 0:
            print(f"{total_failed} documents failed to index.")

        if refresh:
            self.es_client.indices.refresh(index=self.index_name)

    def create_index(self):
        create_index(self.es_client, self.index_name, self.vector_dims)

    def prepare_documents(self, folder_path):
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        nct_id = data.get('nct_id')
                        if nct_id and nct_id not in self.existing_document_ids:
                            documents.append(data)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}: invalid JSON.")
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
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
                    "nct_id": {"type": "keyword"},
                    "condition": {"type": "text"},
                    "condition_vector": {"type": "dense_vector", "dims": vector_dims},  # NEW: Add condition vector
                    "overall_status": {"type": "keyword"},
                    "start_date": {"type": "date", "format": "yyyy-MM-dd"},  
                    "completion_date": {"type": "date", "format": "yyyy-MM-dd"},  
                    "phase": {"type": "keyword"},
                    "study_type": {"type": "keyword"},
                    "condition": {"type": "keyword"},
                    "intervention": {
                        "properties": {
                            "intervention_type": {"type": "keyword"},
                            "intervention_name": {"type": "text"}
                        }
                    },
                    "gender": {"type": "keyword"},
                    "minimum_age": {"type": "keyword"},
                    "maximum_age": {"type": "keyword"},
                    "location": {
                        "properties": {
                            "location_name": {"type": "text"},
                            "location_address": {"type": "text"}
                        }
                    },
                    "reference": {
                        "type": "nested",
                        "properties": {
                            "citation": {"type": "text"},
                            "PMID": {"type": "keyword"}
                        }
                    }
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
        ca_certs="ca.crt",
        basic_auth=(username, password),
        max_retries=50,
        request_timeout=60,
        retry_on_timeout=True
    )

    embedder = SentenceEmbedder() 
    vector_dims = 1024

    folder_path = '../../data/trials_jsons'
    index_name = "clinical_trials"
    indexer = DocumentIndexer(es_client, index_name, embedder, vector_dims)

    while True:
        documents = indexer.prepare_documents(folder_path)
        print(f"Documents prepared: {len(documents)}")
        if documents:
            indexer.index_data(documents, batch_size=500, max_workers=100)
        else:
            print("No new documents to process. Waiting for new files...")
        
        time.sleep(5)  
