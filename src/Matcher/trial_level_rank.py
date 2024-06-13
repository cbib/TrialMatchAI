import pickle
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrialRankingProcessor:
    def __init__(self, inclusion_pickle: str, exclusion_pickle: str):
        self.inclusion_pickle = inclusion_pickle
        self.exclusion_pickle = exclusion_pickle
        self.inclusion_documents = self.load_documents(self.inclusion_pickle)
        self.exclusion_documents = self.load_documents(self.exclusion_pickle)

    def load_documents(self, pickle_file: str) -> List[Dict]:
        with open(pickle_file, 'rb') as f:
            documents = pickle.load(f)
        logger.info("Loaded %d documents from %s", len(documents), pickle_file)
        return documents

    def split_trials(self) -> Tuple[Dict[str, List[Dict]], Dict[str, Tuple[List[Dict], List[Dict]]], List[str]]:
        inclusion_trials = {}
        exclusion_trials = {}

        for doc in self.inclusion_documents:
            nct_id = doc.metadata['_source']['nct_id']
            if nct_id not in inclusion_trials:
                inclusion_trials[nct_id] = []
            inclusion_trials[nct_id].append(doc)

        for doc in self.exclusion_documents:
            nct_id = doc.metadata['_source']['nct_id']
            if nct_id not in exclusion_trials:
                exclusion_trials[nct_id] = []
            exclusion_trials[nct_id].append(doc)

        both_criteria = {nct_id for nct_id in inclusion_trials if nct_id in exclusion_trials}
        only_inclusion = {nct_id for nct_id in inclusion_trials if nct_id not in exclusion_trials}
        only_exclusion = {nct_id for nct_id in exclusion_trials if nct_id not in inclusion_trials}

        return (
            {nct_id: inclusion_trials[nct_id] for nct_id in only_inclusion},
            {nct_id: (inclusion_trials[nct_id], exclusion_trials[nct_id]) for nct_id in both_criteria},
            list(only_exclusion)
        )

    def normalize_scores(self, documents: List[Dict]) -> List[Dict]:
        scores = [doc.metadata['rerank_score'] for doc in documents]
        min_score, max_score = min(scores), max(scores)
        logger.info("Normalizing scores with min: %f, max: %f", min_score, max_score)

        for doc in documents:
            doc.metadata['normalized_score'] = (doc.metadata['rerank_score'] - min_score) / (max_score - min_score + 1e-9)  # Avoid division by zero
        return documents

    def rank_trials(self, trials: Dict[str, List[Dict]], score_key: str = 'normalized_score') -> List[Tuple[str, float, int]]:
        ranked_trials = []
        for nct_id, docs in trials.items():
            weighted_score = sum(doc.metadata[score_key] for doc in docs)
            criteria_count = len(docs)
            ranked_trials.append((nct_id, weighted_score, criteria_count))
        ranked_trials.sort(key=lambda x: x[1], reverse=True)
        return ranked_trials

    def rank_mixed_trials(self, trials: Dict[str, Tuple[List[Dict], List[Dict]]], score_key: str = 'normalized_score') -> List[Tuple[str, float, int, int]]:
        ranked_trials = []
        for nct_id, (inclusion_docs, exclusion_docs) in trials.items():
            inclusion_score = sum(doc.metadata[score_key] for doc in inclusion_docs)
            exclusion_score = sum(doc.metadata[score_key] for doc in exclusion_docs)
            normalized_score = inclusion_score / (exclusion_score + 1e-9)  # Avoid division by zero
            ranked_trials.append((nct_id, normalized_score, len(inclusion_docs), len(exclusion_docs)))
        ranked_trials.sort(key=lambda x: x[1], reverse=True)
        return ranked_trials

    def process(self) -> List[Dict]:
        inclusion_trials, mixed_trials, _ = self.split_trials()

        # Normalize the scores
        self.inclusion_documents = self.normalize_scores(self.inclusion_documents)
        self.exclusion_documents = self.normalize_scores(self.exclusion_documents)

        ranked_inclusion_only = self.rank_trials(inclusion_trials)
        ranked_mixed = self.rank_mixed_trials(mixed_trials)

        # Merge the lists with inclusion-only trials first, followed by mixed trials
        final_ranked_trials = [{'nct_id': nct_id, 'score': score, 'inclusion_count': inc_count, 'exclusion_count': 0} for nct_id, score, inc_count in ranked_inclusion_only] + \
                              [{'nct_id': nct_id, 'score': score, 'inclusion_count': inc_count, 'exclusion_count': exc_count} for nct_id, score, inc_count, exc_count in ranked_mixed]

        logger.info("Final ranked trials: %d", len(final_ranked_trials))
        return final_ranked_trials


if __name__ == "__main__":
    inclusion_pickle_path = 'inclusion.pkl'
    exclusion_pickle_path = 'exclusion.pkl'

    processor = TrialRankingProcessor(inclusion_pickle_path, exclusion_pickle_path)
    final_ranked_trials = processor.process()

    # Save or further process the final ranked trials
    with open('final_ranked_trials.pkl', 'wb') as f:
        pickle.dump(final_ranked_trials, f)
    logger.info("Saved final ranked trials to pickle file")
