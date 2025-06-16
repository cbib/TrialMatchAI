import logging
from typing import List, Dict, Union, Optional, Tuple
from typing import Any
from typing import cast
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime
from dateutil import parser as date_parser
import pandas as pd
from ..Parser.biomedner_engine import BioMedNER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================
# Query Embedding Model
# ========================
class QueryEmbedder:
    def __init__(
        self, model_name="ncbi/MedCPT-Query-Encoder", max_length=512, use_gpu=True
    ):
        """
        Initializes the QueryEmbedder with the specified model and tokenizer.
        """
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def get_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.
        """
        embeddings_dict = {}

        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                embeddings = self.model(**encoded).last_hidden_state[:, 0, :]

            embeddings_dict[text] = embeddings.T.flatten().tolist()

        return embeddings_dict


# ========================
# Clinical Trial Search
# ========================
class ClinicalTrialSearch:
    def __init__(
        self,
        es_client: Elasticsearch,
        embedder: QueryEmbedder,
        index_name: str,
        bio_med_ner: BioMedNER,
    ):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = index_name
        self.bio_med_ner = bio_med_ner 

    def get_synonyms(self, condition: str) -> List[str]:
        raw_result = self.bio_med_ner.annotate_texts_in_parallel(
            [condition], max_workers=1
        )
        ner_results = cast(List[List[Dict[str, Any]]], raw_result)

        if ner_results and ner_results[0]:
            synonyms = set()
            for entity in ner_results[0]:
                if entity.get("entity_group", "").lower() == "disease":
                    synonyms.update(entity.get("synonyms", []))
            return list(synonyms)

        logger.warning(
            f"No annotations found or invalid format for condition: {condition}"
        )
        return []

    def parse_age_input(self, age_input: Union[int, str]) -> Optional[int]:
        """Parse the age input and return the age in years."""
        if isinstance(age_input, int):
            return age_input
        elif isinstance(age_input, str):
            age_input = age_input.strip().lower()
            # Check if input is in the form of "50", "50 years", "50 yrs"
            age_keywords = ["year", "years", "yr", "yrs"]
            try:
                # Attempt to extract age number
                for keyword in age_keywords:
                    if age_input.endswith(keyword):
                        age_str = age_input.replace(keyword, "").strip()
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
                age = (
                    today.year
                    - dob.year
                    - ((today.month, today.day) < (dob.month, dob.day))
                )
                if age < 0:
                    raise ValueError(
                        "Date of birth is in the future. Please provide a valid date of birth."
                    )
                return age
            except (ValueError, OverflowError, date_parser.ParserError):
                pass
        return None

    def get_max_text_score(self, synonyms: List[str]) -> float:
        """
        Estimate the maximum possible text score for normalization.
        We use a 'should' clause for all synonyms, attempting to find
        the highest potential BM25 text match in the index.
        """
        should_clauses = []
        for syn in synonyms:
            should_clauses.append(
                {
                    "multi_match": {
                        "query": syn,
                        "fields": [
                            "condition^6",
                            "eligibility_criteria^4",
                            "brief_title^3",
                            "brief_summary^2",
                            "detailed_description^1.5",
                            "official_title",
                        ],
                        "type": "best_fields",
                        "operator": "and",
                    }
                }
            )

        response = self.es_client.search(
            index=self.index_name,
            body={
                "size": 1,
                "query": {
                    "bool": {"should": should_clauses, "minimum_should_match": 0}
                },
                "track_total_hits": False,
                "sort": [],
                "_source": False,
            },
        )
        max_score = response["hits"]["max_score"]
        logger.info(f"Estimated max text score: {max_score}")
        return max_score if max_score else 1.0  # Avoid division by zero

    def create_query(
        self,
        synonyms: List[str],
        embeddings: Dict[str, List[float]],
        age: int,
        sex: str,
        overall_status: Optional[str],
        max_text_score: float,
        vector_score_threshold: float = 0.5,
        pre_selected_nct_ids: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
    ) -> Dict:
        """
        Builds a hybrid query that combines BM25 text match with
        vector-based similarity using script_score.
        The weighting can be tuned to balance text vs. vector relevance.
        """
        logger.info(
            f"Creating hybrid query for synonyms: {synonyms}, age: {age}, sex: {sex}, overall_status: {overall_status}"
        )

        # Map sex to acceptable gender terms in your index
        sex = sex.upper()
        if sex in ["MALE", "M"]:
            gender_terms = ["MALE", "Male", "male", "M", "All", "all", "ALL"]
        elif sex in ["FEMALE", "F"]:
            gender_terms = ["FEMALE", "Female", "female", "F", "All", "all", "ALL"]
        elif sex in ["ALL", "BOTH"]:
            gender_terms = [
                "All",
                "all",
                "ALL",
                "Both",
                "both",
                "BOTH",
                "FEMALE",
                "Female",
                "female",
                "F",
                "MALE",
                "Male",
                "male",
                "M",
            ]
        else:
            gender_terms = ["All"]

        # Acceptable overall_status values in your index
        acceptable_status = [
            "All",
            "Recruiting",
            "Active, not recruiting",
            "Completed",
            "Terminated",
            "Withdrawn",
            "Not yet recruiting",
            "Suspended",
            "Unknown status",
            "Enrolling by invitation",
        ]
        lowercase_status = [status.lower() for status in acceptable_status]
        if overall_status and overall_status.lower() not in lowercase_status:
            logger.warning(
                f"Invalid overall_status: {overall_status}. Using 'None' instead."
            )
            overall_status = None

        # Build filters
        filters = []

        if age not in [None, "all", "ALL", "All"]:
            filters.append(
                {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": [
                                        {"range": {"minimum_age": {"lte": age}}},
                                        {
                                            "bool": {
                                                "must_not": {
                                                    "exists": {"field": "minimum_age"}
                                                }
                                            }
                                        },
                                        {
                                            "script": {
                                                "script": {
                                                    "source": "return doc['minimum_age'].size() == 0 || doc['minimum_age'].value == null;"
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "should": [
                                        {"range": {"maximum_age": {"gte": age}}},
                                        {
                                            "bool": {
                                                "must_not": {
                                                    "exists": {"field": "maximum_age"}
                                                }
                                            }
                                        },
                                        {
                                            "script": {
                                                "script": {
                                                    "source": "return doc['maximum_age'].size() == 0 || doc['maximum_age'].value == null;"
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                        ]
                    }
                }
            )

        if overall_status and overall_status.lower() != "all":
            filters.append({"match": {"overall_status": overall_status}})

        if pre_selected_nct_ids is not None:
            filters.append({"terms": {"nct_id": pre_selected_nct_ids}})

        if gender_terms:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"terms": {"gender": gender_terms}},
                            {"bool": {"must_not": {"exists": {"field": "gender"}}}},
                            {
                                "script": {
                                    "script": {
                                        "source": "return doc['gender'].size() == 0 || doc['gender'].value == null;"
                                    }
                                }
                            },
                        ]
                    }
                }
            )
        # ============================
        # Text-Based "Should" Clauses
        # ============================
        should_clauses = []
        all_conditions = synonyms
        # Increase or fine-tune the field-level boosts below to emphasize certain fields
        for condition in all_conditions:
            if condition:
                should_clauses.append(
                    {
                        "multi_match": {
                            "query": condition,
                            # Increased boosting for 'condition', 'brief_summary', etc.
                            "fields": [
                                "condition^6",
                                "eligibility_criteria^4",
                                "brief_title^3",
                                "brief_summary^2",
                                "detailed_description",
                                "official_title",
                            ],
                            "type": "best_fields",
                            "operator": "and",
                        },
                    }
                )
                should_clauses.append(
                    {
                        "multi_match": {
                            "query": condition,
                            # Increased boosting for 'condition', 'brief_summary', etc.
                            "fields": [
                                "condition^6",
                                "eligibility_criteria^4",
                                "brief_title^3",
                                "brief_summary^2",
                                "detailed_description^1.5",
                                "official_title",
                            ],
                            "type": "best_fields",
                            "operator": "or",
                        },
                    }
                )
                # Additional phrase search to handle exact matches
                should_clauses.append(
                    {
                        "multi_match": {
                            "query": condition,
                            "fields": [
                                "condition^6",
                                "eligibility_criteria^4",
                                "brief_title^3",
                                "brief_summary^2",
                                "detailed_description^1.5",
                                "official_title",
                            ],
                            "type": "phrase",
                        }
                    }
                )

        # =====================
        # Build the Hybrid Query
        # =====================
        query = {
            "script_score": {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 0,
                        "filter": filters,
                    }
                },
                # -----------------------------------------------------------
                # The scoring script combines both BM25 score ("_score")
                # and vector-based cosine similarity.
                # You can tune these weights:
                #   - alpha for text-based relevance
                #   - beta for vector-based relevance
                # -----------------------------------------------------------
                "script": {
                    "source": """
                        double alpha = 0.5;  // weight for text score
                        double beta  = 0.5;  // weight for vector score

                        double textScore = _score;  // BM25/text relevance
                        double maxTextScore = params.max_text_score;
                        double normalizedTextScore = (maxTextScore == 0) ? 0 : textScore / maxTextScore;

                        double maxConditionVectorScore = 0.0;
                        double maxTitleVectorScore     = 0.0;
                        double maxSummaryVectorScore   = 0.0;
                        double maxEligibilityVectorScore = 0.0;
                        double totalOtherConditionScore = 0.0;

                        // Number of main query vectors
                        int queryVectorCount = params.query_vectors.length;

                        // Calculate maximum similarity for the main query vectors
                        for (int i = 0; i < queryVectorCount; ++i) {
                            maxConditionVectorScore = Math.max(
                                maxConditionVectorScore,
                                cosineSimilarity(params.query_vectors[i], 'condition_vector')
                            );
                            maxTitleVectorScore = Math.max(
                                maxTitleVectorScore,
                                cosineSimilarity(params.query_vectors[i], 'brief_title_vector')
                            );
                            maxSummaryVectorScore = Math.max(
                                maxSummaryVectorScore,
                                cosineSimilarity(params.query_vectors[i], 'brief_summary_vector')
                            );
                            maxEligibilityVectorScore = Math.max(
                               maxEligibilityVectorScore,
                               cosineSimilarity(params.query_vectors[i], 'eligibility_criteria_vector')
                            );
                        }

                        // Calculate total similarity for "other_conditions"
                        int otherConditionCount = params.other_condition_vectors.length;
                        for (int i = 0; i < otherConditionCount; ++i) {
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'condition_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_title_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'eligibility_criteria_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_summary_vector');
                        }

                        // Average out the "other_conditions" vector score
                        if (otherConditionCount > 0) {
                            totalOtherConditionScore /= (otherConditionCount * 4);
                        }

                        // Normalize the vector scores from [-1, +1] to [0, 1]
                        double normalizedConditionScore = (maxConditionVectorScore + 1.0) / 2.0;
                        double normalizedTitleScore = (maxTitleVectorScore + 1.0) / 2.0;
                        double normalizedSummaryScore = (maxSummaryVectorScore + 1.0) / 2.0;
                        double normalizedEligibilityScore = (maxEligibilityVectorScore + 1.0) / 2.0;
                        double normalizedOtherConditionScore = (totalOtherConditionScore + 1.0) / 2.0;

                        // Combine vector scores (weighted average)
                        double combinedVectorScore = (
                            0.3 * normalizedConditionScore + 
                            0.1 * normalizedTitleScore + 
                            0.1 * normalizedSummaryScore + 
                            0.2 * normalizedOtherConditionScore +
                            0.3 * normalizedEligibilityScore
                        );

                        // Threshold check
                        if (combinedVectorScore  < params.vector_score_threshold) {
                            // Exclude document if combined vector score is below threshold
                            return 0;
                        }

                        // Weighted combination of text + vector scores
                        return alpha * normalizedTextScore + beta * combinedVectorScore;
                    """,
                    "params": {
                        "query_vectors": list(embeddings.values()),
                        "other_condition_vectors": (
                            list(
                                self.embedder.get_embeddings(other_conditions).values()
                            )
                            if other_conditions
                            else []
                        ),
                        "max_text_score": max_text_score,
                        "vector_score_threshold": vector_score_threshold,
                    },
                },
            }
        }

        return query

    def search_trials(
        self,
        condition: str,
        age_input: Union[int, str],
        sex: str,
        overall_status: Optional[str] = None,
        size: int = 10,
        pre_selected_nct_ids: Optional[List[str]] = None,
        synonyms: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
        vector_score_threshold: float = 0.0,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Main function to perform the search for clinical trials using a hybrid approach.
        """
        # Parse the age input
        if age_input not in ["all", "ALL", "All"]:
            age = self.parse_age_input(age_input)
        else:
            age = None
        if age is None:
            raise ValueError(
                "Could not parse age input. Please provide a valid age or date of birth."
            )
        logger.info(f"Computed age: {age} years")

        # Combine condition and synonyms
        primary_synonyms = [condition] + (synonyms or [])
        all_terms = primary_synonyms + (other_conditions or [])
        logger.info(f"Using primary synonyms: {primary_synonyms}")
        logger.info(f"Using other conditions: {other_conditions}")

        # ==============================
        # Get embeddings for all terms
        # ==============================
        embeddings = self.embedder.get_embeddings(all_terms)

        # ==============================
        # Get the max text score
        # ==============================
        max_text_score = self.get_max_text_score(primary_synonyms)

        # ==============================
        # Build the query
        # ==============================
        query = self.create_query(
            primary_synonyms,
            embeddings,
            age,
            sex,
            overall_status,
            max_text_score,
            vector_score_threshold=vector_score_threshold,
            pre_selected_nct_ids=pre_selected_nct_ids,
            other_conditions=other_conditions,
        )

        # Execute the search
        response = self.es_client.search(
            index=self.index_name,
            body={
                "size": size,
                "query": query,
            },
        )
        hits = response["hits"]["hits"]

        # Extract the trials and scores
        trials = [hit["_source"] for hit in hits]
        scores = [hit["_score"] for hit in hits]

        # Combine trials and scores into a single list
        trials_with_scores = list(zip(trials, scores))

        # Sort by descending score
        trials_with_scores.sort(key=lambda x: x[1], reverse=True)

        # (Optional) Filter to top 90% (or whichever threshold you deem best)
        top_x_percent_index = int(len(trials_with_scores) * 1.0)
        trials = [trial for trial, score in trials_with_scores[:top_x_percent_index]]

        logger.info(f"Found {len(trials)} trials matching the search criteria.")
        logger.info(f"Top 5 raw scores: {scores[:5]}")
        return trials, scores


########################################
# Example Usage
########################################
if __name__ == "__main__":
    es_url = "https://localhost:9200"
    username = "elastic"
    password = "QQ7wWoB_WnKe*L*X9tAW"

    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        hosts=[es_url],
        ca_certs="certs/ca.crt",
        basic_auth=(username, password),
        request_timeout=300,
        retry_on_timeout=True,
    )

    # Initialize the embedder (QueryEmbedder or FirstLevelSentenceEmbedder)
    # Here we use QueryEmbedder for demonstration
    embedder = QueryEmbedder(model_name="BAAI/bge-m3", max_length=6, use_gpu=False)
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

    # Initialize ClinicalTrialSearch
    index_name = "trec_trials_v1"
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    # Search parameters
    condition = "Hypertrophic Cardiomyopathy"
    age = 23
    sex = "All"
    overall_status = "All"

    # If you already have a list of synonyms:
    main_conditions = [
        "HCM",
        "Asymmetric Septal Hypertrophy",
        "Idiopathic Hypertrophic Subaortic Stenosis",
        "Hypertrophic Obstructive Cardiomyopathy",
    ]

    other_conditions = [
        "Syncope",
        "Exercise-induced syncope",
        "Lightheadedness",
        "Asymmetric interventricular septal hypertrophy",
        "Systolic murmur",
        "Hypertrophic cardiomyopathy",
        "Family history of sudden death",
        "Cardiomyopathy",
        "Exercise intolerance",
        "Arrhythmia",
        "Sudden cardiac death",
        "Heart murmur",
        "Cardiac hypertrophy",
        "Non-obstructive hypertrophic cardiomyopathy",
        "Obstructive hypertrophic cardiomyopathy",
        "Genetic predisposition to cardiovascular disease",
        "Cardiovascular risk factor",
        "Heart disease",
        "Physical activity-related cardiovascular events",
        "Pre-syncope",
        "Ventricular hypertrophy",
        "Cardiac arrhythmia",
        "Sudden death in young adults",
        "Cardiac screening in athletes",
        "Preventive cardiology",
        "Echocardiography findings",
        "Cardiac imaging",
        "Cardiovascular genetics",
        "Cardiovascular health in young adults",
        "Heart failure",
        "Cardiac rehabilitation",
        "Risk assessment in young adults with cardiac symptoms",
    ]

    # Preselected NCT IDs for TREC 2021, for example
    df_trec21 = pd.read_csv("../../data/Unique_NCT_IDs_from_2022_File.csv")
    nct_ids21 = df_trec21["Unique NCT IDs"].unique().tolist()
    pre_selected_nct_ids = nct_ids21

    # Perform the search
    results, scores = cts.search_trials(
        condition,
        age,
        sex,
        overall_status,
        size=round(len(pre_selected_nct_ids) * 0.02),
        pre_selected_nct_ids=pre_selected_nct_ids,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=0.5,
    )

    # Extract relevant fields
    trial_titles = [
        trial.get("brief_title") for trial in results if "brief_title" in trial
    ]
    trial_conditions = [
        trial.get("condition") for trial in results if "condition" in trial
    ]
    nct_ids = [trial.get("nct_id") for trial in results if "nct_id" in trial]

    # Save the retrieved results to a text file
    with open("trial_info.txt", "w") as f:
        for nct_id, title, cond, score in zip(
            nct_ids, trial_titles, trial_conditions, scores
        ):
            f.write(
                f"NCT ID: {nct_id}\nTitle: {title}\nCondition: {cond}\nScore: {score}\n\n"
            )

    # Save only NCT IDs to a separate file
    with open("nct_ids.txt", "w") as f:
        f.write("\n".join(nct_id for nct_id in nct_ids if nct_id is not None))

    # Save NCT IDS with scores
    with open("nct_ids_with_scores.txt", "w") as f:
        for nct_id, score in zip(nct_ids, scores):
            f.write(f"{nct_id}\t{score}\n")

    print("Saved NCT IDs to nct_ids.txt. Number of trials retrieved:", len(results))
