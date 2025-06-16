import logging
from typing import *
from elasticsearch import Elasticsearch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorWithPadding,
)
from functools import partial
from datasets import Dataset
import numpy as np
from collections import defaultdict
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from tqdm import tqdm

from .llm_reranker import LLMReranker
from ..Parser.biomedner_engine import BioMedNER

logger = logging.getLogger(__name__)

NUM_WORKERS = 1
BATCH_SIZE = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster performance


class SecondLevelSentenceEmbedder:
    """
    Generates embeddings for a given sentence using a model like 'BAAI/bge-m3'.
    Can be adapted to utilize GPU if available.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(
            "SecondLevelSentenceEmbedder initialized with model: %s", model_name
        )

    @staticmethod
    def mean_pooling(
        model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, sentence: str) -> List[float]:
        encoded_input = self.tokenizer(
            [sentence], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()


# ---------------- Optional QueryEmbedderSecondLevel Example ---------------- #
class QueryEmbedderSecondLevel:
    """
    Example for generating embeddings using a different model,
    e.g. "ncbi/MedCPT-Query-Encoder".
    """

    def __init__(
        self, model_name="ncbi/MedCPT-Query-Encoder", max_length=10, use_gpu=True
    ):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.model.eval()

    def get_embeddings(self, text: str) -> List[float]:
        """
        Generates embeddings for a given text using the loaded model.
        """
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**encoded)
            # For some models, the [CLS] token embedding is out.last_hidden_state[:, 0, :]
            cls_emb = out.last_hidden_state[:, 0, :]
            cls_emb = F.normalize(cls_emb, p=2, dim=1)
        return cls_emb.flatten().tolist()


def transform_func(
    tokenizer: PreTrainedTokenizerFast, max_length: int, examples: Dict[str, List]
) -> BatchEncoding:
    return tokenizer(
        examples["contents"],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        truncation=True,
    )


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)(
                {k: _move_to_cuda(v) for k, v in maybe_tensor.items()}
            )
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


class RetrievalModel:
    def __init__(self, pretrained_model_name: str, **kwargs):
        self.pretrained_model_name = pretrained_model_name
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.gpu_count = torch.cuda.device_count()
        self.batch_size = BATCH_SIZE

        self.query_instruction = "{}"
        self.document_instruction = "{}"
        self.pool_type = "cls"
        self.max_length = 10

        self.encoder.cuda()
        self.encoder.eval()

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        input_texts = [self.query_instruction.format(s) for s in sentences]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({"contents": input_texts})
        dataset.set_transform(partial(transform_func, self.tokenizer, self.max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
            collate_fn=data_collator,
            pin_memory=True,
        )

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc="encoding", mininterval=10):
            batch_dict = move_to_cuda(batch_dict)
            with torch.amp.autocast("cuda"):
                outputs = self.encoder(**batch_dict)
                if isinstance(
                    outputs, torch.Tensor
                ):  # If model outputs a tensor directly
                    embeddings = outputs  # Direct tensor (batch_size, hidden_dim)
                else:  # If model outputs an object with last_hidden_state
                    embeddings = outputs[0][:, 0, :]  # CLS token pooling
                encoded_embeds.append(embeddings.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


class SecondStageRetriever:
    def __init__(
        self,
        es_client: Elasticsearch,
        llm_reranker_model: LLMReranker,
        embedder: SecondLevelSentenceEmbedder,
        index_name: str,
        size: int = 250,
        inclusion_weight: float = 1.0,
        exclusion_weight: float = 0.25,
        bio_med_ner: BioMedNER = None,
    ):
        self.es_client = es_client
        self.llm_reranker = llm_reranker_model
        self.embedder = embedder
        self.index_name = index_name
        self.size = size
        self.inclusion_weight = inclusion_weight
        self.exclusion_weight = exclusion_weight
        self.bio_med_ner = bio_med_ner

    def get_synonyms(self, condition: str) -> List[str]:
        """
        Use BioMedNER to find synonyms for a given condition.
        """
        ner_results = self.bio_med_ner.annotate_texts_in_parallel(
            [condition], max_workers=1
        )
        if isinstance(ner_results, list):
            synonyms = set()
            for entity in ner_results[0]:
                if (
                    entity["entity_group"].lower() == "disease"
                ):  # Filter for disease entities
                    synonyms.update(entity.get("synonyms", []))
            return list(synonyms)
        else:
            logger.warning(f"No annotations found for condition: {condition}")
            return []

    def retrieve_criteria(
        self, nct_ids: List[str], queries: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve criteria for each query in parallel and return a mapping of query to its retrieved documents.
        """
        query_to_hits = {}

        def execute_query(query):
            # Embed the query
            query_vector = self.embedder.encode_sentences([query])[0]

            # Build the query for Elasticsearch
            es_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["criterion", "entities.synonyms"],
                                        "type": "best_fields",
                                        "operator": "and",
                                    }
                                },
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["criterion", "entities.synonyms"],
                                        "type": "phrase",
                                    }
                                },
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["criterion", "entities.synonyms"],
                                        "type": "best_fields",
                                        "operator": "or",
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                            "filter": {"terms": {"nct_id": nct_ids}},
                        }
                    },
                    "script": {
                        "source": """
                            double alpha = 0.5;
                            double beta = 0.5;

                            double textScore = _score;
                            double vectorScore = (cosineSimilarity(params.query_vector, 'criterion_vector') + 1.0) / 2.0;
                            
                            if (vectorScore < params.vector_score_threshold) {
                                return 0;
                            }
                            
                            return alpha * textScore + beta * vectorScore;
                        """,
                        "params": {
                            "query_vector": query_vector,
                            "vector_score_threshold": 0.5,
                        },
                    },
                }
            }

            # Execute the query
            response = self.es_client.search(
                index=self.index_name,
                body={"size": self.size, "query": es_query},
            )

            hits = response["hits"]["hits"]
            hits = hits[
                : int(len(hits) * 1.0)
            ]  # Top 95% filtering (effectively keeps all results)

            logger.info(f"Retrieved {len(hits)} documents for query: '{query}'")
            return query, hits

        # Use ThreadPoolExecutor to parallelize query execution
        with ThreadPoolExecutor(
            max_workers=min(8, len(queries))
        ) as executor:  # Limit concurrency to 8 threads
            future_to_query = {
                executor.submit(execute_query, query): query for query in queries
            }

            for future in as_completed(future_to_query):
                query, hits = future.result()
                query_to_hits[query] = hits

        return query_to_hits

    def rerank_criteria(self, queries: List[str], criteria: List[Dict]) -> List[Dict]:
        # Create pairs for reranking
        pairs = [
            (criterion["query"], criterion["_source"]["criterion"])
            for criterion in criteria
        ]

        # Perform LLM reranking
        llm_scores = self.llm_reranker.rank_pairs(pairs)

        # Extract scores if they are dictionaries
        if isinstance(llm_scores[0], dict):
            llm_scores = [score.get("llm_score", 0.0) for score in llm_scores]

        # Validate score length
        if len(llm_scores) != len(pairs):
            logger.error("Mismatch between LLM scores and pairs!")
            raise ValueError("Mismatch between LLM scores and pairs!")

        # Assign scores back to criteria
        for i, criterion in enumerate(criteria):
            llm_score = float(llm_scores[i])
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()

            if eligibility_type == "inclusion criteria":
                llm_score *= self.inclusion_weight
            elif eligibility_type == "exclusion criteria":
                llm_score *= self.exclusion_weight

            criterion["llm_score"] = llm_score

        return criteria

    def aggregate_to_trials(
        self, criteria: List[Dict], threshold: float = 0.5, method: str = "weighted"
    ) -> List[Dict]:
        """
        Aggregate individual criterion scores to a trial-level score using different normalization methods.
        This approach provides a flexible way to reward trials that have either a few very high scores
        or many moderately high scores, while preventing trials with many criteria from being over-credited.

        Args:
            criteria (List[Dict]): List of criteria dicts containing scores.
                Each dict is expected to have a '_source' key with an 'nct_id', and an 'llm_score'.
            threshold (float): Minimum llm_score for a criterion to be considered.
            method (str): Aggregation method to use. Options are:
                - 'avg': average score.
                - 'sqrt': sum of scores divided by sqrt(count).
                - 'log': sum of scores divided by log(count + 1).
                - 'weighted': 70% based on sqrt-normalized sum + 30% based on the maximum score.

        Returns:
            List[Dict]: List of dictionaries where each dictionary has 'nct_id' and the aggregated 'score',
                        sorted by descending score.
        """
        # Group scores by trial.
        trial_scores = defaultdict(list)
        for criterion in criteria:
            nct_id = criterion["_source"]["nct_id"]
            score = criterion["llm_score"]
            if score >= threshold:
                trial_scores[nct_id].append(score)

        aggregated_scores = {}
        for nct_id, scores in trial_scores.items():
            count = len(scores)
            total = sum(scores)

            if count == 0:
                continue  # Safety check, though filtering should prevent this.

            if method == "avg":
                agg_score = total / count
            elif method == "sqrt":
                agg_score = total / math.sqrt(count)
            elif method == "log":
                agg_score = total / math.log(count + 1)
            elif method == "weighted":
                # Compute a sqrt-normalized sum.
                sqrt_norm = total / math.sqrt(count)
                # Also consider the maximum score.
                max_score = max(scores)
                # Weighted combination; you can adjust these weights as needed.
                agg_score = 0.7 * sqrt_norm + 0.3 * max_score
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")

            aggregated_scores[nct_id] = agg_score

        # Sort trials by aggregated score (highest first) and return as a list of dicts.
        sorted_trials = [
            {"nct_id": nct_id, "score": score}
            for nct_id, score in sorted(
                aggregated_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return sorted_trials

    def retrieve_and_rank(
        self, queries: List[str], nct_ids: List[str], top_n: int
    ) -> List[Dict]:
        # Retrieve criteria grouped by query
        query_to_hits = self.retrieve_criteria(nct_ids, queries)

        # Flatten all criteria and maintain their associated queries
        all_criteria = []
        for query, hits in query_to_hits.items():
            for hit in hits:
                hit["query"] = query
                all_criteria.append(hit)

        # Rerank the criteria
        ranked_criteria = self.rerank_criteria(queries, all_criteria)

        # Aggregate and rank trials
        sorted_trials = self.aggregate_to_trials(ranked_criteria)

        # Get top N trials
        top_trials = sorted_trials[:top_n]
        logger.info(f"Top {top_n} trials retrieved: {top_trials}")

        with open("top_trials.txt", "w") as f:
            for trial_id, score in top_trials:
                f.write(f"{trial_id}\n")

        logger.info("Top trials saved to top_trials.txt")
        return top_trials


# ========================
# Example
# ========================
if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="certs/ca.crt",
        basic_auth=(username, password),
        max_retries=50,
        request_timeout=120,
        retry_on_timeout=True,
    )

    # Initialize BioMedNER
    bio_med_ner_params = {
        "biomedner_home": "../Parser",
        "biomedner_port": 18894,
        "gner_port": 18783,
        "gene_norm_port": 18888,
        "disease_norm_port": 18892,
        "use_neural_normalizer": True,
        "no_cuda": False,
    }
    bio_med_ner = BioMedNER(**bio_med_ner_params)

    llm_reranker_model_path = "google/gemma-2-2b-it"
    adapter_path = "finetuning/finetune_instruct_gemma2/finetuned_gemma2"

    llm_reranker_model = LLMReranker(
        model_path=llm_reranker_model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.float16,
        batch_size=20,
    )

    embedder = RetrievalModel("BAAI/bge-m3")
    index_name = "trec_trials_eligibility"
    retriever = SecondStageRetriever(
        es_client, llm_reranker_model, embedder, index_name, bio_med_ner=bio_med_ner
    )

    with open("nct_ids.txt", "r") as f:
        nct_ids = [line.strip() for line in f.readlines()]

    queries = [
        "The patient has a diagnosis of anaplastic astrocytoma located in the thoracolumbar spine.",
        "The spinal tumor is classified as unresectable and has been treated with radiation therapy.",
        "The patient experiences severe lower extremity weakness, which has progressed over time.",
        "Urinary retention is present, and the patient is managed with a Foley catheter.",
        "The patient has a history of hypertension that requires ongoing management.",
        "Chronic pain is a significant issue for the patient, impacting daily activities.",
        "An MRI revealed a spinal cord conus mass, which was subsequently biopsied and confirmed as anaplastic astrocytoma.",
        "The patient received field radiation therapy targeting the T10-L1 region of the spine.",
        "Chemotherapy treatment included temozolomide administered in a 7 days on and 7 days off schedule for 11 cycles.",
        "Subsequent chemotherapy involved CPT-11 administered weekly for four weeks, followed by Avastin every two weeks with a two-week rest period before repeating the cycle.",
        "Neurological deficits are evident, characterized by right lower extremity weakness and numbness in the right anterior thigh.",
        "The patient has been evaluated for potential complications related to steroid use.",
    ]

    # Get synonyms for first query and add to the list
    synonyms = retriever.get_synonyms(queries[0])
    queries.extend(synonyms)

    top_trials = retriever.retrieve_and_rank(queries, nct_ids, top_n=260)

    print("Top Trials:")
    for trial_id, score in top_trials:
        print(f"NCT ID: {trial_id}, Score: {score}")
