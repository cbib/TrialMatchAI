from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


PoolingStrategy = Literal["cls", "mean"]


@dataclass(frozen=True)
class TextEmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    pooling: PoolingStrategy = "mean"
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = True
    use_fp16: bool = False
    normalize: bool = True


class TextEmbedder:
    def __init__(self, config: TextEmbedderConfig):
        self.config = config
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif config.use_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(
            "Loading embedder model %s on device %s", config.model_name, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()
        if config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Embedder model converted to FP16")

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise ValueError("Cannot embed empty text.")
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return []
        vectors: List[List[float]] = []
        for batch in _batched(cleaned, self.config.batch_size):
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.config.max_length,
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**enc)
                pooled = self._pool(outputs, enc["attention_mask"])
                if self.config.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
