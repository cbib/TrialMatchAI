import logging
from typing import List, Dict, Union, Callable, Sequence
import elasticsearch
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from joblib import Parallel, delayed
import pickle
import json
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentence Embedding Model
class SentenceEmbedder:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        logger.info("SentenceEmbedder initialized with model %s", model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentence: str) -> List[float]:
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        logger.info("Generated embeddings for sentence: %s", sentence)
        return sentence_embeddings[0].tolist()

embedder = SentenceEmbedder()

class ClinicalTrialSearch:
    def __init__(self, es_client: Elasticsearch, embedder: SentenceEmbedder, index_name: str):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = index_name
        # Gender mapping (same as before)
        self.gender_mapping = {
            "MALE": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "Male": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "male": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "M": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "m": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "FEMALE": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "Female": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "female": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "F": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "f": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "OTHER_SEX": ["All", "Other", "Unknown", "all", "ALL"],
            "UNKNOWN_SEX": ["All", "Other", "Unknown", "all", "ALL"],
            "Other": ["All", "Other", "Unknown", "all", "ALL"],
            "Unknown": ["All", "Other", "Unknown", "all", "ALL"]
        }

    def create_must_query(self, condition: str, age: int, sex: str, overall_status: str) -> Dict:
        """Create a must query for BM25 search based on condition, age, sex, and overall_status."""
        # Map the provided sex to possible gender values in the clinical_trials index
        gender_terms = self.gender_mapping.get(sex, ["All", "Unknown", "all", "ALL"])
        print(f"Using gender terms: {gender_terms}")
        
        logger.info(f"Creating BM25 must query for condition: {condition}, age: {age}, sex: {sex}, overall_status: {overall_status}")
        
        # Gender queries as a list of should conditions, one for each gender term
        gender_query = [{"match": {"gender": term}} for term in gender_terms]
        
        return {
            "bool": {
                "must": [
                    {"range": {"minimum_age": {"lte": age}}},  # Minimum age condition
                    {"match": {"overall_status": overall_status}}  # Overall status match (e.g., Recruiting)
                ],
                "filter": [
                    {"bool": {
                        "should": gender_query,  # Gender matching conditions
                        "minimum_should_match": 1  # Ensure at least one gender matches
                    }},
                    {
                        "bool": {
                            "should": [
                                {"range": {"maximum_age": {"gte": age}}},  # Case when maximum_age is present
                                {"bool": {"must_not": {"exists": {"field": "maximum_age"}}}}  # Case when maximum_age is null
                            ],
                            "minimum_should_match": 1  # Ensure at least one age condition is satisfied
                        }
                    }
                ]
            }
        }

    def scroll_search_trials(self, condition: str, age: int, sex: str, overall_status: str, scroll_time: str = '2m', batch_size: int = 1000) -> List[str]:
        """
        Perform a scroll search to retrieve all trials that match the condition, age, sex, and overall status.
        This function uses the scroll API to retrieve large datasets in batches.
        
        Args:
        - condition (str): The medical condition to search for (e.g., "Colorectal cancer").
        - age (int): The age of the patient to check eligibility.
        - sex (str): The sex of the patient (e.g., "MALE", "FEMALE").
        - overall_status (str): The recruitment status (e.g., "Recruiting").
        - scroll_time (str): The time the scroll context remains active (default '2m' - 2 minutes).
        - batch_size (int): The number of documents to retrieve in each batch (default is 1000).
        
        Returns:
        - List[str]: A list of nct_ids of all trials matching the search criteria.
        """
        
        # Build the query
        search_query = {
            "query": self.create_must_query(condition, age, sex, overall_status),
            "size": batch_size
        }
        
        # Perform the initial search with scroll
        response = self.es_client.search(
            index=self.index_name,
            body=search_query,
            scroll=scroll_time
        )
        
        # Initialize scroll_id and collect the first batch of hits
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        # Collect the nct_ids from the first batch
        all_nct_ids = [hit["_source"]["nct_id"] for hit in hits]
        
        # Continue scrolling until there are no more results
        while len(hits) > 0:
            response = self.es_client.scroll(
                scroll_id=scroll_id,
                scroll=scroll_time
            )
            hits = response['hits']['hits']
            all_nct_ids.extend([hit["_source"]["nct_id"] for hit in hits])  # Collect more nct_ids
        
        # Clear the scroll context to free resources
        self.es_client.clear_scroll(scroll_id=scroll_id)
        
        logger.info(f"Found {len(all_nct_ids)} trials from clinical_trials search using scroll.")
        return all_nct_ids

# Elasticsearch Retriever Class for Eligibility Criteria
class ElasticsearchRetriever:
    def __init__(self, es_client: Elasticsearch, index_name: Union[str, Sequence[str]], 
                 body_func: Callable[[str], Dict], content_field: str):
        self.es_client = es_client
        self.index_name = index_name
        self.body_func = body_func
        self.content_field = content_field

    def search_query(self, query: str, scroll: str = '10m', size: int = 5000) -> List[Dict]:
        body = self.body_func(query)
        body['size'] = size
        results = []
        response = self.es_client.search(index=self.index_name, body=body, scroll=scroll)
        scroll_id = response.get('_scroll_id')
        while True:
            hits = response['hits']['hits']
            if not hits:
                break
            results.extend(hits)
            response = self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)
        self.es_client.clear_scroll(scroll_id=scroll_id)
        logger.info(f"Total results for query '{query}': {len(results)}")
        return [{"_source": hit["_source"], "_id": hit["_id"], "_score": hit["_score"]} for hit in results]

    def normal_search_query(self, query: str, size: int = 3000) -> List[Dict]:
        body = self.body_func(query)
        body['size'] = size
        response = self.es_client.search(index=self.index_name, body=body)
        results = response['hits']['hits']
        logger.info(f"Total results for query '{query}': {len(results)}")
        return [{"_source": hit["_source"], "_id": hit["_id"], "_score": hit["_score"]} for hit in results]

# Three Stage Search for Eligibility Criteria
class ThreeStageSearch:
    def __init__(self, es_client: Elasticsearch, embedder, n_jobs: int = 10):
        self.es_client = es_client
        self.embedder = embedder
        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, device=6)
        self.index_name = "eligibility_criteria"
        self.text_field = "criterion"
        self.n_jobs = n_jobs
        self.top_k_initial = 5000  # Top documents before reranking

    def reciprocal_rank_fusion(self, results: List[Dict], k: int = 60) -> List[Dict]:
        """Combine search results using Reciprocal Rank Fusion (RRF)."""
        combined_scores = {}
        for result_set in results:
            source = result_set['source']
            for rank, result in enumerate(result_set['results']):
                doc_id = result['id']
                score = 1 / (k + rank + 1)
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = 0
                combined_scores[doc_id] += score
        combined_results = [{'id': doc_id, 'score': score} for doc_id, score in combined_scores.items()]
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Unique document IDs after RRF: {len(combined_results)}")
        return combined_results
    
    def query_body_text(self, search_query: str, nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> Dict:
        """Create a text query for searching the eligibility criteria index."""
        logger.info(f"Constructing text query body for nct_ids: {nct_ids}")
        return {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"nct_id": nct_ids}},
                        {"match": {"eligibility_type.keyword": eligibility_type}}
                    ],
                    "should": [
                        {
                            "match": {
                                "criterion": {
                                    "query": search_query,
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

    def query_body_vector(self, search_query: str, nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> Dict:
        """Create a vector-based search query."""
        vector = self.embedder.get_embeddings(search_query)
        logger.info(f"Constructing vector query body for nct_ids: {nct_ids}")
        return {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"nct_id": nct_ids}},
                        {"match": {"eligibility_type.keyword": eligibility_type}}
                    ],
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc['criterion_vector']) + 1.0",
                                    "params": {"query_vector": vector}
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

    def retrieve_combined_query(self, search_queries: List[str], nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Document]:
        """Retrieve documents using text and vector queries with RRF combination."""
        text_results = []
        vector_results = []
        for search_query in search_queries:
            text_retriever = ElasticsearchRetriever(
                es_client=self.es_client,
                index_name=self.index_name,
                body_func=lambda query: self.query_body_text(query, eligibility_type, nct_ids),
                content_field=self.text_field
            )
            vector_retriever = ElasticsearchRetriever(
                es_client=self.es_client,
                index_name=self.index_name,
                body_func=lambda query: self.query_body_vector(query, eligibility_type, nct_ids),
                content_field=self.text_field
            )
            text_results.extend(text_retriever.normal_search_query(search_query))
            vector_results.extend(vector_retriever.normal_search_query(search_query))

        rrf_results = self.reciprocal_rank_fusion([
            {"source": "text", "results": text_results},
            {"source": "vector", "results": vector_results}
        ])
        rrf_doc_ids = [res['id'] for res in rrf_results]

        final_documents = [Document(page_content=doc["_source"][self.text_field], metadata=doc) 
                           for doc in text_results + vector_results if doc["_id"] in rrf_doc_ids]
        return final_documents

# Functions to handle Phenopacket
def load_phenopacket(json_file: str) -> Dict:
    """Load the phenopacket from a JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def parse_phenopacket(phenopacket: Dict) -> Dict:
    """Extract relevant data from the phenopacket."""
    parsed_data = {
        "sex": phenopacket["subject"]["sex"],
        "birth_date": phenopacket["subject"]["date_of_birth"],
        "diseases": [disease["term"]["label"] for disease in phenopacket["diseases"]],
        "phenotypic_features": [f["type"]["label"] for f in phenopacket.get("phenotypic_features", [])],
        "medical_history": [mh["term"]["label"] for mh in phenopacket.get("subject", {}).get("medical_history", [])],
        "genomic_interpretations": [f"Gene: {gi['gene_descriptor']['symbol']}, Mutation: {gi['variation_descriptor']['hgvs']}"
                                    for gi in phenopacket.get("genomic_interpretations", [])],
        "medical_actions": [f"{ma['procedure']['code']['label']} performed on {ma['procedure']['performed']['timestamp']}"
                            if "procedure" in ma else f"Treatment: {ma['treatment']['agent']['label']}"
                            for ma in phenopacket.get("medical_actions", [])]
    }
    return parsed_data

def calculate_age(birth_date: str) -> int:
    """Calculate the patient's age from the birth date."""
    birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
    return (datetime.now() - birth_date).days // 365

# Main Execution
if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="ca.crt",
        basic_auth=(username, password),
        request_timeout=60,
        retry_on_timeout=True
    )

    # Load and parse phenopacket
    phenopacket_file_path = "../../data/synthetic_patients/phenopacket1.json"
    phenopacket = load_phenopacket(phenopacket_file_path)
    parsed_data = parse_phenopacket(phenopacket)

    # Extract condition, age, sex for clinical_trials search
    condition = parsed_data["diseases"][0]
    birth_date = parsed_data["birth_date"]
    sex = parsed_data["sex"]
    age = calculate_age(birth_date)

    # Clinical trials search to get nct_ids
    clinical_trial_search = ClinicalTrialSearch(es_client, embedder, index_name="clinical_trials")
    nct_ids = clinical_trial_search.scroll_search_trials(condition="colon cancer", age=54, sex= "male", overall_status="Completed", scroll_time = '6m', batch_size = 1000)

    # # Formulate search queries for eligibility_criteria using parsed phenopacket fields
    # search_queries = parsed_data["diseases"] + parsed_data["phenotypic_features"] + parsed_data["medical_history"] + \
    #                  parsed_data["genomic_interpretations"] + parsed_data["medical_actions"]

    # # Eligibility criteria search with full phenopacket on filtered nct_ids
    # three_stage_search = ThreeStageSearch(es_client, embedder, n_jobs=2)
    # eligibility_results = three_stage_search.retrieve_combined_query(search_queries, nct_ids, "Inclusion Criteria")

    # # Process and log results
    # logger.info(f"Total eligibility results found: {len(eligibility_results)}")

    es_client.close()
    logger.info("Closed Elasticsearch client")