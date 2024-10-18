import logging
from typing import List, Dict, Union, Callable, Sequence, Any
import elasticsearch
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from FlagEmbedding import FlagReranker, FlagLLMReranker
from langchain_core.documents import Document
import pickle
import asyncio
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_reranker import Reranker
import concurrent.futures
import os

# Assign GPUs 0 and 1 as visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,6,7"


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

    def scroll_search_query(self, query: str, scroll: str = '1m') -> List[Dict]:
        # Modify the body to include only 'nct_id' and 'criterion'
        body = self.body_func(query)
        body["_source"] = ["nct_id", "criterion", "criteria_id"]  # Only fetch 'nct_id' and 'criterion'
        results = []
        
        # Execute the search query
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

    def normal_search_query(self, query: str, size: int = 10000) -> List[Dict]:
        # Modify the body to include only 'nct_id' and 'criterion'
        body = self.body_func(query)
        body['size'] = size
        body["_source"] = ["nct_id", "criterion", "criteria_id"]  # Only fetch 'nct_id' and 'criterion'
        
        # Execute the search query
        response = self.es_client.search(index=self.index_name, body=body)
        results = response['hits']['hits']
        
        logger.info("Total results for query '%s': %d", query, len(results))
        
        return [{"_source": hit["_source"], "query": query, "_id": hit["_id"], "_score": hit["_score"]} for hit in results]

    async def normal_search(self, queries: List[str], size: int = 10000) -> List[Dict]:
        # Run synchronous Elasticsearch searches in separate threads using asyncio.to_thread
        tasks = [asyncio.to_thread(self.normal_search_query, query, size) for query in queries]
        all_results = await asyncio.gather(*tasks)
        return [item for sublist in all_results for item in sublist]
    
    async def scroll_search(self, queries: List[str], scroll: str = '1m') -> List[Dict]:
        # Run synchronous Elasticsearch searches in separate threads using asyncio.to_thread
        tasks = [asyncio.to_thread(self.scroll_search_query, query, scroll) for query in queries]
        all_results = await asyncio.gather(*tasks)
        return [item for sublist in all_results for item in sublist]


class ThreeStageSearch:
    def __init__(self, es_client: Elasticsearch, embedder, reranker_model: str = 'BAAI/bge-reranker-v2-gemma', adapter_path: str = "./finetuned_reranker"):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = "eligibility_criteria_v2"
        self.text_field = "criterion"
        self.first_level_reranker = FlagReranker('BAAI/bge-reranker-v2-m3', device=None, use_fp16=True)
        self.llm_reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', peft_path=adapter_path, device=5, use_fp16=True)
        self.adapter_path = adapter_path
        logger.info("ThreeStageSearch initialized")


    def reciprocal_rank_fusion(self, results: List[Dict], k: int = 60) -> List[Dict]:

        combined_scores = defaultdict(float)  # Store document IDs with accumulated scores

        def process_result_set(result_set):
            # Get document IDs and assign ranks based on their order in the list
            doc_ids = np.array([result['id'] for result in result_set['results']])
            ranks = np.arange(1, len(result_set['results']) + 1)  # Ranks starting from 1
            scores = 1 / (k + ranks)  # RRF score: 1 / (k + rank)
            return doc_ids, scores

        # Parallel processing using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_result_set, result_set) for result_set in results]

            # Aggregate scores across all futures
            for future in futures:
                doc_ids, scores = future.result()
                for doc_id, score in zip(doc_ids, scores):
                    combined_scores[doc_id] += score

        # Prepare combined results: list of dictionaries with doc_id and its final score
        combined_results = [{'id': doc_id, 'score': score} for doc_id, score in combined_scores.items()]

        # Sort documents by the combined score in descending order
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        return combined_results

    def query_body_text(self, search_query: str, nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> Dict:
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
                        {"match": {"eligibility_type.keyword": eligibility_type}},
                        {"terms": {"nct_id": nct_ids}}  # Filter by nct_ids
                    ],
                    "should": should_clauses,
                    "minimum_should_match": 1  # At least one should clause should match
                }
            }
        }

    def query_body_vector(self, search_query: str, nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> Dict:
        vector = embedder.get_embeddings(search_query)

        logger.info("Constructed vector query body for query: %s", search_query)
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"eligibility_type.keyword": eligibility_type}},
                        {"terms": {"nct_id": nct_ids}}  # Filter by nct_ids
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

    async def retrieve_text_query(self, search_queries: List[str], nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Dict]:
        logger.info("Starting text retrieval for queries")
        text_retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.index_name,
            body_func=lambda query: self.query_body_text(query, nct_ids, eligibility_type),
            content_field=self.text_field,
        )

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                text_documents = await text_retriever.normal_search(search_queries)
                logger.info("The number of text documents retrieved: %d", len(text_documents))
                return [{"id": doc["_id"], "score": doc["_score"], "query": doc["query"], "_source": doc["_source"]} for doc in text_documents]
            except elasticsearch.ApiError as e:
                logger.error("Text retrieval attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_attempts:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    async def retrieve_vector_query(self, search_queries: List[str], nct_ids: List[str], eligibility_type: str = "Inclusion Criteria") -> List[Dict]:
        logger.info("Starting vector retrieval for queries")
        vector_retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.index_name,
            body_func=lambda query: self.query_body_vector(query, nct_ids, eligibility_type),
            content_field=self.text_field,
        )

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                vector_documents = await vector_retriever.normal_search(search_queries, size=5000)
                logger.info("The number of vector documents retrieved: %d", len(vector_documents))
                return [{"id": doc["_id"], "score": doc["_score"], "query": doc["query"], "_source": doc["_source"]} for doc in vector_documents]
            except elasticsearch.ApiError as e:
                logger.error("Vector retrieval attempt %d failed: %s", attempt + 1, e)
                if attempt < retry_attempts:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    async def retrieve_combined_query(
        self,
        search_queries: List[str],
        nct_ids: List[str],
        eligibility_type: str = "Inclusion Criteria"
    ) -> List[Document]:
        logger.info("Starting retrieval for combined query")

        # Retrieve text and vector query results
        text_results = await self.retrieve_text_query(search_queries, nct_ids, eligibility_type)
        sorted_text_results = sorted(text_results, key=lambda x: x["score"], reverse=True)
        sorted_text_results = [doc for doc in sorted_text_results if doc["score"] > 5]
        vector_results = await self.retrieve_vector_query(search_queries, nct_ids, eligibility_type)
        sorted_vector_results = sorted(vector_results, key=lambda x: x["score"], reverse=True)
        sorted_vector_results = [doc for doc in sorted_vector_results if doc["score"] > 3.2]
        
        logger.info("Retrieved %d text documents and %d vector documents",
                    len(text_results), len(vector_results))

        # Combine deduplicated text and vector results
        combined_results = sorted_vector_results + sorted_text_results
        # Deduplication based on 'query', '_source.criteria_id', and '_source.criterion'
        seen = set()
        deduplicated_results = []

        for item in combined_results:
            unique_key = (item['query'], item['_source']['criteria_id'], item['_source']['criterion'])
            
            if unique_key not in seen:
                deduplicated_results.append(item)
                seen.add(unique_key)
                
        logger.info("Combined deduplicated documents: %d", len(deduplicated_results))
        
        # Prepare pairs for reranking
        pairs = [
            [doc["query"], doc["_source"][self.text_field]]
            for doc in deduplicated_results
        ]
        
        first_reranker_scores = self.first_level_reranker.compute_score(pairs, normalize=True, batch_size=50)
        
        torch.cuda.empty_cache()
    
        logger.info("Computed first level reranker scores")

        for i, doc in enumerate(deduplicated_results):
            doc["rerank_score"] = first_reranker_scores[i]

        positive_documents = [doc for doc in deduplicated_results if doc["rerank_score"] > 0.5]
        sorted_documents = sorted(positive_documents, key=lambda x: x["rerank_score"], reverse=True)
        
        pairs = [
            (doc["query"], doc["_source"][self.text_field])
            for doc in sorted_documents
        ]

        # Compute reranker scores asynchronously
        llm_reranker_scores = self.llm_reranker.compute_score(pairs, batch_size=5, prompt = "Analyze the medical condition described in query A and the statement in passage B. Determine if the statement is in accordance with the condition. Respond with 'Yes' if they are suitable and aligned or 'No' if otherwise.")
        

        # Assign rerank scores to documents
        for doc, score in zip(sorted_documents, llm_reranker_scores):
            doc["llm_rerank_score"] = score

        # Create final Document instances
        final_documents = [
            Document(page_content=doc["_source"][self.text_field], metadata=doc)
            for doc in sorted_documents
        ]

        return final_documents

if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="certs/ca.crt",
        basic_auth=(username, password),
        max_retries=50,
        request_timeout=60,
        retry_on_timeout=True
    )

    logger.info("Testing Elasticsearch connection")

    async def main():
        info = es_client.info()
        logger.info("Connected to Elasticsearch: %s", info)

        embedder = SentenceEmbedder()
        three_stage_search = ThreeStageSearch(es_client=es_client, embedder=embedder)

        search_queries = [
                    "Histologically confirmed non-small cell lung cancer (NSCLC)",
                    "Confirmed EGFR C797S mutation.",
                    "ECOG performance status 1 (able to carry out light work)",
                    "Patient with a measurable CNS progression based on RECIST 1.1 criteria.",
                    "New or progressive brain metastases confirmed by imaging.",
                    "Treated with first-line EGFR tyrosine kinase inhibitors (e.g., osimertinib) in a previous treatment.",
                    "No other limitations on previous treatments.",
                    "Central nervous system (CNS) involvement: Positive CSF cytology confirming leptomeningeal metastases.",
                    "Measurable disease: Brain metastases measurable based on RECIST 1.1 criteria.",
                    "Hemoglobin ≥9.0 g/dL", 
                    "leukocytes ≥3.0 x 10^9/L", 
                    "Absolute neutrophil count ≥1.5 x 10^9/L", 
                    "platelets ≥100 x 10^9/L.",
                    "Liver function: AST, ALT ≤3x upper limit of normal (ULN), total bilirubin ≤1.5x ULN.",
                    "Kidney function: Creatinine ≤1.5x ULN, or eGFR ≥60 mL/min.",
                    "Pregnancy status: Post-menopausal, not pregnant, and not breastfeeding.",
                    "Contraception: Agrees to use contraception for the duration of the study and for 3 months after the last dose of the study drug.",
                    "Stable on no more than 2 mg dexamethasone per day with no recent steroid dose adjustments.",
                    "Not on enzyme-inducing anticonvulsants for at least 2 weeks prior to enrollment.",
                    "No recent cardiac history of prolonged QT interval, torsades de pointes, or family history of long QT syndrome.",
                    "No prior treatments with anti-EGFR therapies (erlotinib, gefitinib, afatinib, osimertinib) within 14 days of registration.",
                    "No prior biologic therapy.",
                    "No evidence of active hepatitis B, hepatitis C, or HIV infection.",
                    "Willing and able to sign informed consent and return for regular follow-ups.",
                    "Life expectancy of at least 3 months, based on physician’s assessment.",
                    "Willing to provide mandatory blood specimens for correlative research."
                ]
                        
        def load_nct_ids(file_path: str) -> List[str]:
            with open(file_path, 'r') as file:
                # Read each line, strip newlines and extra spaces, and store in a list
                nct_ids = [line.strip() for line in file.readlines()]
            return nct_ids

        # Usage example:
        nct_ids_file = 'nct_ids.txt'
        nct_ids = load_nct_ids(nct_ids_file)
        print(len(nct_ids))

        logger.info("Starting search for queries: %s", search_queries)
        inclusion_documents = await three_stage_search.retrieve_combined_query(search_queries, nct_ids=nct_ids, eligibility_type="Inclusion Criteria")
        torch.cuda.empty_cache()
        exclusion_documents = await three_stage_search.retrieve_combined_query(search_queries, nct_ids=nct_ids, eligibility_type="Exclusion Criteria")

        with open('inclusion.pkl', 'wb') as f:
            pickle.dump(inclusion_documents, f)
        logger.info("Saved inclusion documents to pickle file")
        
        with open('exclusion.pkl', 'wb') as f:
            pickle.dump(exclusion_documents, f)
        logger.info("Saved exclusion documents to pickle file")

        es_client.close()
        logger.info("Closed Elasticsearch client")

    asyncio.run(main()) 
