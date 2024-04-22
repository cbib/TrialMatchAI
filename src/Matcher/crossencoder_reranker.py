import json
import logging
from typing import List

import torch
from sagemaker_inference import encoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PAIRS = "pairs"
SCORES = "scores"


class CrossEncoder:
    def __init__(self) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")
        model_name = "BAAI/bge-reranker-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    def __call__(self, pairs: List[List[str]]) -> List[float]:
        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = inputs.to(self.device)
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        return scores.detach().cpu().tolist()


def model_fn(model_dir: str) -> CrossEncoder:
    try:
        return CrossEncoder()
    except Exception:
        logging.exception(f"Failed to load model from: {model_dir}")
        raise


def transform_fn(
    cross_encoder: CrossEncoder, input_data: bytes, content_type: str, accept: str
) -> bytes:
    payload = json.loads(input_data)
    model_output = cross_encoder(**payload)
    output = {SCORES: model_output}
    return encoder.encode(output, accept)