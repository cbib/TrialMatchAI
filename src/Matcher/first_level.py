import logging
from typing import List, Dict, Union, Optional
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datetime import datetime
from dateutil import parser as date_parser

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
        token_embeddings = model_output[0]  # First element is the token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def get_embeddings(self, sentences: List[str]) -> Dict[str, List[float]]:
        # Generate embeddings for each sentence (condition and synonyms)
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings = {sentence: embedding.tolist() for sentence, embedding in zip(sentences, sentence_embeddings)}
        logger.info("Generated embeddings for sentences: %s", sentences)
        return embeddings

class ClinicalTrialSearch:
    def __init__(self, es_client: Elasticsearch, embedder: SentenceEmbedder, index_name: str):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = index_name

    def parse_age_input(self, age_input: Union[int, str]) -> Optional[int]:
        """Parse the age input and return the age in years."""
        if isinstance(age_input, int):
            # Direct age input
            return age_input
        elif isinstance(age_input, str):
            age_input = age_input.strip().lower()
            # Check if input is in the form of "50", "50 years", "50 yrs"
            age_keywords = ["year", "years", "yr", "yrs"]
            try:
                # Attempt to extract age number
                for keyword in age_keywords:
                    if age_input.endswith(keyword):
                        age_str = age_input.replace(keyword, '').strip()
                        age = int(age_str)
                        return age
                # Try to parse as integer
                age = int(age_input)
                return age
            except ValueError:
                pass
            # Attempt to parse as date of birth
            try:
                dob = date_parser.parse(age_input, fuzzy=True)
                today = datetime.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                if age < 0:
                    raise ValueError("Date of birth is in the future. Please provide a valid date of birth.")
                return age
            except (ValueError, OverflowError, date_parser.ParserError):
                pass
        # If all parsing attempts fail
        return None

    # Remove this method from the class
    def get_synonyms(self, condition: str) -> List[str]:
        pass  # This method is no longer needed


    def get_max_text_score(self, synonyms: List[str]) -> float:
        """Estimate the maximum possible text score for normalization."""
        # Build should clauses for each synonym
        should_clauses = []
        for syn in synonyms:
            should_clauses.append({
                "multi_match": {
                    "query": syn,
                    "fields": [
                        "condition^5",
                        "brief_title^3",
                        "brief_summary^2",
                        "detailed_description^2"
                    ],
                    "type": "best_fields",
                    "operator": "or"
                }
            })

        response = self.es_client.search(
            index=self.index_name,
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "track_total_hits": False,
                "sort": [],
                "_source": False
            }
        )
        max_score = response['hits']['max_score']
        logger.info(f"Estimated max text score: {max_score}")
        return max_score if max_score else 1.0  # Avoid division by zero

    def create_query(self, synonyms: List[str], embeddings: Dict[str, List[float]], age: int, sex: str,
                    overall_status: Optional[str], max_text_score: float) -> Dict:
        logger.info(f"Creating query for synonyms: {synonyms}, age: {age}, sex: {sex}, overall_status: {overall_status}")

        # Map sex to gender terms in index
        sex = sex.upper()
        if sex in ["MALE", "M"]:
            gender_terms = ["MALE", "Male", "male", "M", "All", "all", "ALL"]
        elif sex in ["FEMALE", "F"]:
            gender_terms = ["FEMALE", "Female", "female", "F", "All", "all", "ALL"]
        else:
            gender_terms = ["All", "Other", "Unknown", "all", "ALL"]

        # Build filters
        filters = [
            {"range": {"minimum_age": {"lte": age}}},
            {
                "bool": {
                    "should": [
                        {"range": {"maximum_age": {"gte": age}}},
                        {"bool": {"must_not": {"exists": {"field": "maximum_age"}}}}
                    ]
                }
            },
            {"terms": {"gender": gender_terms}}
        ]

        # Conditionally add the overall_status filter
        if overall_status and overall_status.lower() != "all":
            filters.append({"match": {"overall_status": overall_status}})

        # Build should clauses for each synonym
        should_clauses = []
        for syn in synonyms:
            # Phrase search clauses
            should_clauses.append({
                "multi_match": {
                    "query": syn,
                    "fields": [
                        "condition^10",
                        "brief_title^8",
                        "brief_summary^3",
                        "detailed_description^2"
                    ],
                    "type": "phrase"
                }
            })
            # Term search clauses with 'and' operator
            should_clauses.append({
                "multi_match": {
                    "query": syn,
                    "fields": [
                        "condition^5",
                        "brief_title^4",
                        "brief_summary^1",
                        "detailed_description^1"
                    ],
                    "type": "best_fields",
                    "operator": "and"
                }
            })

        # Build the query
        query = {
            "script_score": {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1,
                        "filter": filters
                    }
                },
                "script": {
                    "source": """
                        double textScore = _score;
                        double maxTextScore = params.max_text_score;
                        double normalizedTextScore = textScore / maxTextScore;

                        // Compute max vector score across all synonyms
                        double maxVectorScore = 0.0;
                        for (int i = 0; i < params.query_vectors.length; ++i) {
                            double vectorScore = cosineSimilarity(params.query_vectors[i], 'condition_vector');
                            if (vectorScore > maxVectorScore) {
                                maxVectorScore = vectorScore;
                            }
                        }
                        double normalizedVectorScore = (maxVectorScore + 1.0) / 2.0;

                        // Adjust weights: balance text relevance and vector similarity
                        return (0.3 * normalizedTextScore) + (0.7 * normalizedVectorScore);
                    """,
                    "params": {
                        "query_vectors": list(embeddings.values()),
                        "max_text_score": max_text_score
                    }
                }
            }
        }
        return query


    def search_trials(self, condition: str, age_input: Union[int, str], sex: str,
                    overall_status: Optional[str] = None, size: int = 10,
                    synonyms: Optional[List[str]] = None) -> List[Dict]:
        # Parse the age input to get the age in years
        age = self.parse_age_input(age_input)
        if age is None:
            raise ValueError("Could not parse age input. Please provide a valid age or date of birth.")
        logger.info(f"Computed age: {age} years")

        # Combine the condition and synonyms
        synonyms_with_condition = [condition] + (synonyms or [])
        logger.info(f"Using synonyms for condition '{condition}': {synonyms_with_condition}")

        # Get embeddings for all synonyms
        embeddings = self.embedder.get_embeddings(synonyms_with_condition)

        # Get the max text score for normalization
        max_text_score = self.get_max_text_score(synonyms_with_condition)

        # Build the query
        query = self.create_query(synonyms_with_condition, embeddings, age, sex, overall_status, max_text_score)

        # Execute the search
        response = self.es_client.search(
            index=self.index_name,
            body={
                "size": size,
                "query": query
            }
        )

        hits = response['hits']['hits']
        trials = [hit['_source'] for hit in hits]
        logger.info(f"Found {len(trials)} trials matching the search criteria.")
        return trials


if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="certs/ca.crt",
        basic_auth=(username, password),
        request_timeout=60,
        retry_on_timeout=True
    )

    # Initialize the embedder
    embedder = SentenceEmbedder(model_name='BAAI/bge-m3')

    # List of synonyms for the condition
    condition_synonyms = [
        'lung carcinoma',
        'lung neoplasm',
        'lung tumor',
        'lung tumour',
        'pulmonary cancer',
        'pulmonary carcinoma',
        'small cell lung cancer',
        'non-small cell lung cancer',
        'NSCLC',
        'SCLC'
    ]

    # Initialize the ClinicalTrialSearch
    index_name = "clinical_trials"
    cts = ClinicalTrialSearch(es_client, embedder, index_name)

    # Search parameters
    condition = "non-small cell lung cancer"
    age = "25 years"  # Can be an integer, age string, or date of birth
    sex = "Male"
    overall_status = "Recruiting"  # Can be "Recruiting", "Active", "Not yet recruiting", etc.

    # Perform the search
    results = cts.search_trials(condition, age, sex, overall_status, size=10000, synonyms=condition_synonyms)

    # Extract the NCT IDs from the results
    nct_ids = [trial.get('nct_id') for trial in results if 'nct_id' in trial]

    # Save the list of NCT IDs into a text file
    with open('nct_ids.txt', 'w') as f:
        f.write('\n'.join(nct_ids))

    # Optionally, you can print the NCT IDs to verify
    print("Saved NCT IDs to nct_ids.txt:")
    for nct_id in nct_ids:
        print(nct_id)

