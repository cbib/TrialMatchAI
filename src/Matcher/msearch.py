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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ElasticsearchRetriever:
    def __init__(self, es_client: Elasticsearch, index_name: Union[str, Sequence[str]], 
                 body_func: Callable[[str], Dict], content_field: str):
        self.es_client = es_client
        self.index_name = index_name
        self.body_func = body_func
        self.content_field = content_field

    # msearch function using Elasticsearch's multi-search API to return separate results with a larger size
    def msearch(self, queries: List[str], size: int = 1000) -> List[Dict]:
        search_body = []
        for query in queries:
            search_body.append({"index": self.index_name})
            search_body.append(self.body_func(query, size))

        response = self.es_client.msearch(body=search_body)

        # Separate the results for each query
        results = []
        for query, query_response in zip(queries, response['responses']):
            hits = query_response['hits']['hits']
            results.append({
                "query": query,
                "results": [{"_id": hit["_id"], "_source": hit["_source"], "_score": hit["_score"]} for hit in hits]
            })

        logger.info("Total results for multi-search: %d queries processed", len(results))
        return results

class ThreeStageSearch:
    def __init__(self, es_client: Elasticsearch, embedder, n_jobs: int = 10):
        self.es_client = es_client
        self.embedder = embedder
        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, device=6)
        self.index_name = "eligibility_criteria"
        self.text_field = "criterion"
        self.n_jobs = n_jobs
        self.top_k_initial = 5000  # Number of top documents to select initially before reranking
        logger.info("ThreeStageSearch initialized with n_jobs=%d", n_jobs)
        
    def reciprocal_rank_fusion(self, results: List[Dict], k: int = 60) -> List[Dict]:
        combined_scores = {}
        
        for result_set in results:
            source = result_set['source']
            for rank, result in enumerate(result_set['results']):
                doc_id = result['_id']
                score = 1 / (k + rank + 1)  # RRF score
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = 0
                combined_scores[doc_id] += score

        combined_results = [{'id': doc_id, 'score': score} for doc_id, score in combined_scores.items()]
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Debugging: Log unique document IDs
        unique_ids = [doc['id'] for doc in combined_results]
        logger.info("Unique document IDs after RRF: %d", len(unique_ids))
        logger.debug("Unique document IDs: %s", unique_ids)
        
        return combined_results

    def query_body_text(self, search_query: str, size: int = 1000, eligibility_type: str = "Inclusion Criteria") -> Dict:
        should_clauses = [
            {
                "match": {
                    "criterion": {
                        "query": search_query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            {
                "nested": {
                    "path": "entities",
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "entities.synonyms": {
                                            "query": search_query,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "entities.entity": {
                                            "query": search_query,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

        logger.info("Constructed text query body for query: %s", search_query)
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"eligibility_type.keyword": eligibility_type}}
                    ],
                    "should": should_clauses,
                    "minimum_should_match": 1  # At least one should clause should match
                }
            },
            "size": size  # Ensure we fetch more than the default 10 results
        }

    # Using msearch for text queries and separating results for each query
    def retrieve_text_query(self, search_queries: List[str], eligibility_type: str = "Inclusion Criteria", size: int = 1000) -> List[Dict]:
        logger.info("Starting text retrieval for queries")
        text_retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.index_name,
            body_func=lambda query, size: self.query_body_text(query, size, eligibility_type),
            content_field=self.text_field,
        )

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                # Using msearch instead of normal search with specified size
                text_documents = text_retriever.msearch(search_queries, size)
                logger.info("The number of text documents retrieved: %d", len(text_documents))
                return text_documents  # Returns results for each query separately
            except elasticsearch.ApiError as e:
                logger.error("Text retrieval attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_attempts:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def retrieve_combined_query(self, search_queries: List[str], eligibility_type: str = "Inclusion Criteria", size: int = 1000) -> List[Document]:
        logger.info("Starting retrieval for combined query")
        text_results = self.retrieve_text_query(search_queries, eligibility_type, size)
        
        # Combine results
        rrf_results = []
        for query_result in text_results:
            rrf_results.append({
                "source": "text",
                "results": query_result["results"]
            })

        rrf_combined_results = self.reciprocal_rank_fusion(rrf_results)
        rrf_doc_ids = [res['id'] for res in rrf_combined_results]

        # Filter top N documents based on RRF scores
        all_results = [doc for sublist in text_results for doc in sublist["results"]]
        rrf_top_documents = [doc for doc in all_results if doc["_id"] in rrf_doc_ids]
        rrf_top_documents = sorted(rrf_top_documents, key=lambda x: x['_score'], reverse=True)[:self.top_k_initial]
        logger.info("The number of top documents after RRF: %d", len(rrf_top_documents))

        pairs = [[doc["query"], doc["_source"][self.text_field]] for doc in rrf_top_documents]

        scores = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.reranker.compute_score)([pair]) for pair in pairs)

        for i, doc in enumerate(rrf_top_documents):
            doc["rerank_score"] = scores[i]
        positive_documents = [doc for doc in rrf_top_documents if doc["rerank_score"][0] > 0]
        sorted_documents = sorted(positive_documents, key=lambda x: x["rerank_score"], reverse=True)

        final_documents = [Document(page_content=doc["_source"][self.text_field], metadata=doc) for doc in sorted_documents]
        logger.info("Finished retrieval and ranking for combined query")
        
        return final_documents

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

    logger.info("Testing Elasticsearch connection")
    info = es_client.info()
    logger.info("Connected to Elasticsearch: %s", info)

    embedder = SentenceEmbedder()
    three_stage_search = ThreeStageSearch(es_client=es_client, embedder=embedder, n_jobs=10)

    search_queries = [
    'Cancer Type: Non-Small Cell Lung Cancer (NSCLC)',
    'Cancer Stage: Stage IV (confirmed via histological analysis)',
    'Primary Tumor Site: Right lung',
    # 'Molecular/Genetic Information: Positive for PD-L1 expression (TPS > 1%), EGFR/ALK/ROS-1 mutation negative',
    # 'Performance Status (P.S): ECOG 1, still able to carry out light work',
    # 'Systemic Therapy: First-line therapy',
    # 'Completed 4 cycles of platinum-based doublet chemotherapy (due to absence of driver mutations)',
    # 'Achieved partial response (PR) based on RECIST 1.1 criteria',
    # 'Radiotherapy: No history of prior lung irradiation',
    # 'Candidate for both stereotactic body radiation therapy (SBRT) and low dose radiotherapy (LDRT)',
    # 'Comorbidities: Hypertension (controlled, current BP: 140/90 mmHg), COPD (managed, no interstitial lung disease)',
    # 'Cardiac Health: No history of coronary artery disease, bypass surgery, or severe heart dysfunction (New York Heart Association grade I)',
    # 'Psychiatric History: No severe psychotic symptoms',
    # 'Genetic Mutations and Variations: EGFR/ALK/ROS-1 mutations negative (confirmed), PD-L1 status: Positive (TPS > 1%)',
    # 'Imaging Results:',
    # 'CT Scan (Chest): Multiple lung metastases, no malignant pleural or pericardial effusion',
    # 'Bone Scan: No bone metastases detected',
    # 'MRI (Brain): No evidence of brain metastases',
    # 'Laboratory Measures:',
    # 'Total Bilirubin: 1.2 mg/dL (normal)',
    # 'AST/ALT: AST 30 U/L, ALT 32 U/L (within normal range)',
    # 'WBC: 5,000/uL',
    # 'ANC: 2,000/uL',
    # 'Platelet Count: 110,000/uL',
    # 'Hemoglobin: 10.5 g/dL',
    # 'Creatinine: 1.2 mg/dL (normal)',
    # 'Calcium (Ca2+): 1.3 mmol/L (normal)',
    # 'Past Treatments: Completed 4 cycles of platinum-based doublet chemotherapy with partial response',
    # 'No prior immunotherapy',
    # 'No prior radiotherapy targeting the lungs',
    # 'Exclusion History: No history of interstitial lung disease, active pneumonitis, or autoimmune disease',
    # 'No prior malignancies that could interfere with current cancer treatment',
    # 'No active infection (HIV, Hepatitis B/C, tuberculosis)',
    # 'No uncontrolled hypercalcemia',
    # 'Pregnancy Status: Not applicable (male)',
    # 'Outcomes:',
    # 'Current Status: Partial response after first-line chemotherapy, stable disease (SD)',
    # 'Life Expectancy: Estimated > 6 months'
    ]

    logger.info("Starting search for queries: %s", search_queries)
    documents = three_stage_search.retrieve_combined_query(search_queries, eligibility_type="Inclusion Criteria", size=1000)

    with open('inclusion.pkl', 'wb') as f:
        pickle.dump(documents, f)
    logger.info("Saved documents to pickle file")

    es_client.close()
    logger.info("Closed Elasticsearch client")
