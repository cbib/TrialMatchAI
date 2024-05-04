import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
import argparse
import os
import torch

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def get_dictionaries_of_specific_entities(list_of_dicts, key, values):
    return [d for d in list_of_dicts if any(val in d.get(key, []) for val in values)]

class BioMedNERtmvar:
    def __init__(self, params):
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.model_path_or_name, model_max_length=512, max_length=512, truncation=True)
        self.ner_pipeline = pipeline("ner", model=self.params.model_path_or_name, tokenizer=self.tokenizer, aggregation_strategy="first", device=self.params.device)

    def recognize(self, text):
        result_entities = self.ner_pipeline(text)
        return result_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioMedNERtmvar")
    parser.add_argument("--model_path_or_name", type=str, help="Path to the model")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use")
    args = parser.parse_args()

    ner_model = BioMedNERtmvar(args)
    # Example usage
    text = "p.Leu307Arg"
    entities = ner_model.recognize(text)
    print(entities)

