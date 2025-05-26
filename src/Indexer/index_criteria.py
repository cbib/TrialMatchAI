#!/usr/bin/env python3
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CriteriaIndexer:
    def __init__(self, es: Elasticsearch, index_name: str, processed_file: Path):
        self.es = es
        self.index_name = index_name
        self.processed_file = processed_file
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids = self._load_processed_ids()

    def _load_processed_ids(self) -> set[str]:
        if self.processed_file.exists():
            return set(self.processed_file.read_text().splitlines())
        return set()

    def _save_processed_ids(self):
        self.processed_file.write_text("\n".join(sorted(self.processed_ids)) + "\n")

    def _trial_indexed(self, nct_id: str) -> bool:
        try:
            res = self.es.count(
                index=self.index_name,
                body={"query": {"term": {"nct_id": nct_id}}}
            )
            return res.get("count", 0) > 0
        except NotFoundError:
            return False

    def _detect_dim(self, processed_folder: Path) -> int:
        # find first JSON under any trial subfolder
        for trial_dir in processed_folder.iterdir():
            if not trial_dir.is_dir():
                continue
            for f in trial_dir.glob("*.json"):
                doc = json.loads(f.read_text())
                vec = doc.get("criterion_vector", [])
                if isinstance(vec, list):
                    return len(vec)
        raise RuntimeError("No criterion_vector found in any processed JSON.")

    def create_index(self, dims: int):
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "standard_lowercase": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "criteria_id": {"type": "keyword"},
                    "criterion": {"type": "text", "analyzer": "standard_lowercase"},
                    "criterion_vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw", "m": 16, "ef_construction": 100
                        }
                    },
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "normalized_id": {"type": "keyword"},
                            "synonyms": {"type": "text", "analyzer": "standard_lowercase"},
                            "entity": {"type": "text", "analyzer": "standard_lowercase"},
                            "class": {"type": "keyword"}
                        }
                    },
                    "nct_id": {"type": "keyword"},
                    "eligibility_type": {"type": "keyword"}
                }
            }
        }
        self.es.indices.create(
            index=self.index_name,
            body=mapping,
            timeout="60s",
            master_timeout="60s"
        )
        logger.info(f"Created index {self.index_name} with dims={dims}")

    def _index_trial(self, trial_dir: Path, batch_size: int) -> tuple[str,int]:
        nct_id = trial_dir.name

        # skip if already done or in ES
        if nct_id in self.processed_ids or self._trial_indexed(nct_id):
            logger.info(f"Skipping {nct_id}: already indexed")
            return nct_id, 0

        # load docs
        docs = []
        for f in trial_dir.glob("*.json"):
            try:
                docs.append(json.loads(f.read_text()))
            except Exception as e:
                logger.warning(f"{nct_id}: failed to load {f.name}: {e}")

        if not docs:
            logger.info(f"No JSONs for {nct_id}; marking done")
            return nct_id, 0

        # bulk‐index in sub‐batches
        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            actions = [
                {"_op_type": "index",
                 "_index": self.index_name,
                 "_id": doc["criteria_id"],
                 "_source": doc}
                for doc in batch
            ]
            succ, fails = bulk(
                client=self.es,
                actions=actions,
                raise_on_error=False,
                chunk_size=batch_size
            )
            total += succ
            logger.info(f"{nct_id}: batch {i//batch_size+1} → {succ} indexed, {len(fails)} failed")

        return nct_id, total

    def index_all(self, processed_folder: Path, batch_size: int = 100, max_workers: int = 4, refresh: bool = True):
        trials = [d for d in processed_folder.iterdir() if d.is_dir()]
        if not trials:
            logger.info("No trial subfolders found.")
            return

        # 1) determine vector dims & create index if missing
        if not self.es.indices.exists(index=self.index_name):
            dims = self._detect_dim(processed_folder)
            self.create_index(dims)
        else:
            logger.info(f"Index {self.index_name} exists; skipping creation")

        # 2) parallel indexing across trials
        total_indexed = 0
        newly_done = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_nct = {
                executor.submit(self._index_trial, td, batch_size): td.name
                for td in trials
            }
            for future in as_completed(future_to_nct):
                nct = future_to_nct[future]
                try:
                    nct_id, count = future.result()
                    total_indexed += count
                    newly_done.append(nct_id)
                except Exception as e:
                    logger.error(f"{nct}: unexpected error: {e}")
                    newly_done.append(nct)

        # 3) optionally refresh & persist processed IDs
        if refresh:
            self.es.indices.refresh(index=self.index_name)

        self.processed_ids.update(newly_done)
        self._save_processed_ids()
        logger.info(f"✅ Indexed {total_indexed} criteria across {len(newly_done)} trials (skipped: {len(self.processed_ids)-len(newly_done)}).")


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())

def make_es_client(cfg: dict) -> Elasticsearch:
    es_conf = cfg["elasticsearch"]
    return Elasticsearch(
        hosts=es_conf["hosts"],
        basic_auth=(es_conf["username"], es_conf["password"]),
        ca_certs=es_conf["ca_certs"],
        verify_certs=True,
        request_timeout=es_conf.get("request_timeout", 60),
        max_retries=es_conf.get("max_retries", 3),
        retry_on_timeout=True,
    )

def main():
    parser = argparse.ArgumentParser(description="Bulk‑index prepared eligibility criteria in parallel")
    parser.add_argument("--config",           required=True, help="Path to config.json")
    parser.add_argument("--processed-folder", required=True, help="Root folder of trial subfolders")
    parser.add_argument("--index-name",       default="trec_trials_eligibility_v3", help="ES index name")
    parser.add_argument("--batch-size",       type=int, default=100, help="Docs per bulk request")
    parser.add_argument("--max-workers",      type=int, default=4,   help="Parallel trial threads")
    args = parser.parse_args()

    cfg = load_config(args.config)
    es = make_es_client(cfg)

    indexer = CriteriaIndexer(
        es=es,
        index_name=args.index_name,
        processed_file=Path("processed_trials.txt")
    )
    indexer.index_all(
        processed_folder=Path(args.processed_folder),
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()