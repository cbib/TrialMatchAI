from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, List, Literal, Sequence, Any

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


EmbedderBackend = Literal["hf", "hashing"]
PoolingStrategy = Literal["cls", "mean"]


@dataclass(frozen=True)
class TextEmbedderConfig:
    backend: EmbedderBackend = "hf"
    model_name: str = "BAAI/bge-m3"
    revision: str | None = None
    trust_remote_code: bool = False
    pooling: PoolingStrategy = "mean"
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = True
    use_fp16: bool = False
    normalize: bool = True
    hashing_dimensions: int = 64


class HashingTextEmbedder:
    """Deterministic lightweight embedder for tests and CPU smoke runs."""

    def __init__(self, dimensions: int = 64, normalize: bool = True):
        if dimensions <= 0:
            raise ValueError("hashing_dimensions must be positive.")
        self.dimensions = dimensions
        self.normalize = normalize
        logger.info(
            "Using deterministic hashing embedder with %d dimensions.", dimensions
        )

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise ValueError("Cannot embed empty text.")
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts if text and text.strip()]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        if self.normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
        return vector


class TextEmbedder:
    def __init__(self, config: TextEmbedderConfig):
        torch, F, AutoModel, AutoTokenizer = _load_embedding_dependencies()
        self._torch = torch
        self._functional = F
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading embedder model %s on device %s", config.model_name, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        ).to(self.device)
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
            with self._torch.inference_mode():
                outputs = self.model(**enc)
                pooled = self._pool(outputs, enc["attention_mask"])
                if self.config.normalize:
                    pooled = self._functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors

    def _pool(self, outputs: Any, attention_mask: Any) -> Any:
        if self.config.pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = self._torch.sum(token_embeddings * mask, dim=1)
        counts = self._torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_embedder(config: dict) -> TextEmbedder:
    """Construct a TextEmbedder from the ``embedder`` section of a config dict.

    Single source for the embedder wiring previously copy-pasted across main.py
    and the index/build-concepts/update-registry CLIs.
    """
    embedder_cfg = config.get("embedder", {}) or {}
    backend = embedder_cfg.get("backend", "hf")
    if backend == "hashing":
        return HashingTextEmbedder(
            dimensions=int(embedder_cfg.get("hashing_dimensions", 64)),
            normalize=bool(embedder_cfg.get("normalize", True)),
        )
    if backend != "hf":
        raise ValueError(f"Unsupported embedder backend: {backend}")
    return TextEmbedder(
        TextEmbedderConfig(
            backend="hf",
            model_name=embedder_cfg.get("model_name", "BAAI/bge-m3"),
            revision=embedder_cfg.get("revision"),
            trust_remote_code=embedder_cfg.get("trust_remote_code", False),
            pooling=embedder_cfg.get("pooling", "mean"),
            max_length=embedder_cfg.get("max_length", 512),
            batch_size=embedder_cfg.get("batch_size", 32),
            use_gpu=embedder_cfg.get("use_gpu", True),
            use_fp16=embedder_cfg.get("use_fp16", False),
            normalize=embedder_cfg.get("normalize", True),
        )
    )


def _load_embedding_dependencies():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Text embedding requires the ML dependencies. Install them with "
            "`uv sync --extra llm` or `pip install 'trialmatchai[llm]'`."
        ) from exc
    return torch, F, AutoModel, AutoTokenizer
