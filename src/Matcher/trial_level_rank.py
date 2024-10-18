import pickle
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrialRankingProcessor:
    def __init__(self, inclusion_pickle: str, exclusion_pickle: str):
        self.inclusion_documents = self.load_documents(inclusion_pickle)
        self.exclusion_documents = self.load_documents(exclusion_pickle)

    def load_documents(self, pickle_file: str) -> List[Dict]:
        with open(pickle_file, 'rb') as f:
            documents = pickle.load(f)
        logger.info("Loaded %d documents from %s", len(documents), pickle_file)
        return documents

    def process(self) -> List[Dict]:
        trials = {}

        # Define weights and parameters
        w_inclusion = 1.0
        w_exclusion = 0.5
        amplification_factor = 1.5
        conflict_factor = 0.5
        inclusion_threshold = 0.5
        exclusion_threshold = -0.5

        # Process inclusion documents
        for doc in self.inclusion_documents:
            nct_id = doc.metadata['_source']['nct_id']
            llm_score = doc.metadata['llm_rerank_score']
            confidence = doc.metadata.get('confidence', 1.0)  # Default confidence is 1.0

            weighted_score = llm_score * confidence * w_inclusion

            if nct_id not in trials:
                trials[nct_id] = {'inclusion_scores': [], 'exclusion_scores': []}
            trials[nct_id]['inclusion_scores'].append(weighted_score)

        # Process exclusion documents
        for doc in self.exclusion_documents:
            nct_id = doc.metadata['_source']['nct_id']
            llm_score = doc.metadata['llm_rerank_score']
            confidence = doc.metadata.get('confidence', 1.0)  # Default confidence is 1.0

            # Adjust exclusion score
            if llm_score < 0:
                # Negative score indicates contradiction with exclusion criteria (good)
                adjusted_score = abs(llm_score) * confidence * amplification_factor * w_exclusion
            else:
                # Positive score indicates match with exclusion criteria (bad)
                adjusted_score = -llm_score * confidence * w_exclusion

            if nct_id not in trials:
                trials[nct_id] = {'inclusion_scores': [], 'exclusion_scores': []}
            trials[nct_id]['exclusion_scores'].append(adjusted_score)

        # Separate trials into two groups
        trials_without_exclusion = []
        trials_with_exclusion = []

        for nct_id, scores in trials.items():
            inclusion_score = sum(scores['inclusion_scores'])
            exclusion_score = sum(scores['exclusion_scores'])
            inclusion_count = len(scores['inclusion_scores'])
            exclusion_count = len(scores['exclusion_scores'])

            # Apply conflict penalty
            conflict_penalty = conflict_factor * min(inclusion_score, abs(exclusion_score))
            total_score = inclusion_score + exclusion_score - conflict_penalty

            # Apply thresholds
            if inclusion_score < inclusion_threshold or exclusion_score < exclusion_threshold:
                continue  # Skip this trial

            trial_data = {
                'nct_id': nct_id,
                'total_score': total_score,
                'inclusion_score': inclusion_score,
                'exclusion_score': exclusion_score,
                'conflict_penalty': conflict_penalty,
                'inclusion_count': inclusion_count,
                'exclusion_count': exclusion_count
            }

            if exclusion_count == 0:
                # Trials with no exclusion criteria matches
                trials_without_exclusion.append(trial_data)
            else:
                # Trials with one or more exclusion criteria matches
                trials_with_exclusion.append(trial_data)

        # Rank trials within each group based on total_score
        trials_without_exclusion.sort(key=lambda x: x['total_score'], reverse=True)
        trials_with_exclusion.sort(key=lambda x: x['total_score'], reverse=True)

        # Combine the two groups, ensuring trials without exclusion matches are ranked first
        ranked_trials = trials_without_exclusion + trials_with_exclusion

        logger.info("Final ranked trials: %d", len(ranked_trials))
        return ranked_trials

if __name__ == "__main__":
    inclusion_pickle_path = 'inclusion.pkl'
    exclusion_pickle_path = 'exclusion.pkl'

    processor = TrialRankingProcessor(inclusion_pickle_path, exclusion_pickle_path)
    final_ranked_trials = processor.process()

    # Save or further process the final ranked trials
    with open('final_ranked_trials.pkl', 'wb') as f:
        pickle.dump(final_ranked_trials, f)
    logger.info("Saved final ranked trials to pickle file")
