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
    def __init__(self, model_name: str = 'dmlls/all-mpnet-base-v2-negation'):
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
        logger.info("Total results for query '%s': %d", query, len(results))
        
        return [{"_source": hit["_source"], "query": query, "_id": hit["_id"], "_score": hit["_score"]} for hit in results]

    def normal_search_query(self, query: str, size: int = 3000) -> List[Dict]:
        body = self.body_func(query)
        body['size'] = size
        
        response = self.es_client.search(index=self.index_name, body=body)
        results = response['hits']['hits']
        
        logger.info("Total results for query '%s': %d", query, len(results))
        
        return [{"_source": hit["_source"], "query": query, "_id": hit["_id"], "_score": hit["_score"]} for hit in results]

    def scroll_search(self, queries: List[str], scroll: str = '10m', size: int = 5000, n_jobs: int = 1) -> List[Dict]:
        all_results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.search_query)(query, scroll, size) for query in queries
        )
        return [item for sublist in all_results for item in sublist]

    def normal_search(self, queries: List[str], size: int = 2000, n_jobs: int = 5) -> List[Dict]:
        all_results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.normal_search_query)(query, size) for query in queries
        )
        return [item for sublist in all_results for item in sublist]

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
                doc_id = result['id']
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

    def query_body_text(self, search_query: str, eligibility_type: str = "Inclusion Criteria") -> Dict:
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
                                }
                            ],
                            "must": [
                                {
                                    "match": {
                                        "entities.is_negated": {
                                            "query": "yes",
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
            }
        }

    def query_body_vector(self, search_query: str, eligibility_type: str = "Inclusion Criteria") -> Dict:
        vector = embedder.get_embeddings(search_query)

        logger.info("Constructed vector query body for query: %s", search_query)
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"eligibility_type.keyword": eligibility_type}}
                    ],
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'criterion_vector') + 1.0",
                                    "params": {"query_vector": vector}
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1  # At least one should clause should match
                }
            }
        }

    def retrieve_text_query(self, search_queries: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Dict]:
        logger.info("Starting text retrieval for queries")
        text_retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.index_name,
            body_func=lambda query: self.query_body_text(query, eligibility_type),
            content_field=self.text_field,
        )

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                text_documents = text_retriever.scroll_search(search_queries, n_jobs=self.n_jobs)
                logger.info("The number of text documents retrieved: %d", len(text_documents))
                return [{"id": doc["_id"], "score": doc["_score"], "query": doc["query"], "_source": doc["_source"]} for doc in text_documents]
            except elasticsearch.ApiError as e:
                logger.error("Text retrieval attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def retrieve_vector_query(self, search_queries: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Dict]:
        logger.info("Starting vector retrieval for queries")
        vector_retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.index_name,
            body_func=lambda query: self.query_body_vector(query, eligibility_type),
            content_field=self.text_field,
        )

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                vector_documents = vector_retriever.normal_search(search_queries, n_jobs=self.n_jobs)
                logger.info("The number of vector documents retrieved: %d", len(vector_documents))
                return [{"id": doc["_id"], "score": doc["_score"], "query": doc["query"], "_source": doc["_source"]} for doc in vector_documents]
            except elasticsearch.ApiError as e:
                logger.error("Vector retrieval attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def retrieve_combined_query(self, search_queries: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Document]:
        logger.info("Starting retrieval for combined query")
        text_results = self.retrieve_text_query(search_queries, eligibility_type)
        vector_results = self.retrieve_vector_query(search_queries, eligibility_type)
        
        # Apply RRF to combine results
        rrf_results = self.reciprocal_rank_fusion([
            {"source": "text", "results": text_results},
            {"source": "vector", "results": vector_results}
        ])
        rrf_doc_ids = [res['id'] for res in rrf_results]

        # Filter top N documents based on RRF scores
        all_results = text_results + vector_results
        rrf_top_documents = [doc for doc in all_results if doc["id"] in rrf_doc_ids]
        rrf_top_documents = sorted(rrf_top_documents, key=lambda x: x['score'], reverse=True)[:self.top_k_initial]
        logger.info("The number of top documents after RRF: %d", len(rrf_top_documents))

        pairs = [[doc["query"], doc["_source"][self.text_field]] for doc in rrf_top_documents]

        scores = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.reranker.compute_score)([pair]) for pair in pairs)

        for i, doc in enumerate(rrf_top_documents):
            doc["rerank_score"] = scores[i]

        positive_documents = [doc for doc in rrf_top_documents if doc["rerank_score"] > 0]
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
        ca_certs="certs/ca/ca.crt",
        basic_auth=(username, password),
        max_retries=50,
        request_timeout=60,
        retry_on_timeout=True
    )

    logger.info("Testing Elasticsearch connection")
    info = es_client.info()
    logger.info("Connected to Elasticsearch: %s", info)

    embedder = SentenceEmbedder()
    three_stage_search = ThreeStageSearch(es_client=es_client, embedder=embedder, n_jobs=2)

    search_queries = ["Bone Fracture", "KRAS mutation", "Hypertension"]
     
    logger.info("Starting search for queries: %s", search_queries)
    documents = three_stage_search.retrieve_combined_query(search_queries, eligibility_type="Inclusion Criteria")

    with open('documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    logger.info("Saved documents to pickle file")

    es_client.close()
    logger.info("Closed Elasticsearch client")
