#!/usr/bin/env python3
import os
import json
import re
import argparse
import hashlib
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────────────
class SentenceEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Converted model to FP16")

    def mean_pool(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        summed = torch.sum(token_embeds * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Batch‑embed a list of strings; returns a list of float vectors.
        """
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc)
        vecs = self.mean_pool(outputs.last_hidden_state, enc.attention_mask)
        vecs = F.normalize(vecs, p=2, dim=1)
        return vecs.cpu().tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_criteria_id(nct_id: str, criterion: str) -> str:
    """
    Deterministically hash a trial‑criterion pair to a 64‑hex string.
    """
    return hashlib.sha256(f"{nct_id}:{criterion}".encode("utf-8")).hexdigest()


def load_raw_trial(path: Path) -> dict:
    return json.loads(path.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────
def process_trial(
    nct_id: str,
    source_folder: Path,
    processed_folder: Path,
    embedder: SentenceEmbedder,
) -> int:
    raw_path = source_folder / f"{nct_id}.json"
    if not raw_path.exists():
        logger.warning(f"Missing raw JSON for {nct_id}, skipping.")
        return 0

    data = load_raw_trial(raw_path)
    criteria = data.get("criteria", [])
    if not criteria:
        logger.info(f"No criteria found for {nct_id}.")
        return 0

    # collect texts
    entries = []
    texts = []
    for crit in criteria:
        text = crit.get("criterion") or crit.get("sentence")
        if not text:
            continue
        entries.append({
            "nct_id": nct_id,
            "criterion": text,
            "entities": crit.get("entities", []),
            "eligibility_type": crit.get("type"),
        })
        texts.append(text)

    if not entries:
        return 0

    # embed all at once
    vectors = embedder.embed(texts)

    # write out
    trial_folder = processed_folder / nct_id
    trial_folder.mkdir(parents=True, exist_ok=True)

    for entry, vec in zip(entries, vectors):
        crit_id = compute_criteria_id(entry["nct_id"], entry["criterion"])
        out = {
            "criteria_id": crit_id,
            "nct_id": entry["nct_id"],
            "criterion": entry["criterion"],
            "entities": entry["entities"],
            "eligibility_type": entry["eligibility_type"],
            "criterion_vector": vec,
        }
        (trial_folder / f"{crit_id}.json").write_text(json.dumps(out, indent=2))

    logger.info(f"Processed {len(entries)} criteria for {nct_id}")
    return len(entries)


def main():
    p = argparse.ArgumentParser(description="Prepare & embed eligibility criteria per trial")
    p.add_argument(
        "--ids-file", required=True,
        help="Path to nct_ids.txt (one NCT ID per line)"
    )
    p.add_argument(
        "--source-folder", required=True,
        help="Folder containing raw trial JSONs named <NCT_ID>.json"
    )
    p.add_argument(
        "--processed-folder", default="processed_criteria",
        help="Output root; will contain one subfolder per trial"
    )
    p.add_argument(
        "--model-name", default="BAAI/bge-m3",
        help="Sentence embedding model name"
    )
    p.add_argument(
        "--use-gpu", action="store_true",
        help="Enable GPU iff available"
    )
    args = p.parse_args()

    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    source_folder = Path(args.source_folder)
    processed_folder = Path(args.processed_folder)
    processed_folder.mkdir(parents=True, exist_ok=True)

    embedder = SentenceEmbedder(model_name=args.model_name, use_gpu=args.use_gpu)

    total = 0
    skipped = 0

    for nct in ids:
        trial_folder = processed_folder / nct
        # Skip if already processed (i.e. folder exists and contains at least one .json)
        if trial_folder.exists() and any(trial_folder.glob("*.json")):
            logger.info(f"Skipping {nct}: already processed")
            skipped += 1
            continue

        processed_count = process_trial(nct, source_folder, processed_folder, embedder)
        total += processed_count

    logger.info(f"✅ Finished embedding. Total criteria written: {total}. Trials skipped: {skipped}.")


if __name__ == "__main__":
    main()
