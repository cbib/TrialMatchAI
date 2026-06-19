from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    cfg = json.loads(config_path.read_text())
    es_conf = cfg.setdefault("elasticsearch", {})
    if os.getenv("TRIALMATCHAI_ES_HOST"):
        es_conf["hosts"] = [os.environ["TRIALMATCHAI_ES_HOST"]]
    if os.getenv("TRIALMATCHAI_ES_USERNAME"):
        es_conf["username"] = os.environ["TRIALMATCHAI_ES_USERNAME"]
    if os.getenv("TRIALMATCHAI_ES_PASSWORD"):
        es_conf["password"] = os.environ["TRIALMATCHAI_ES_PASSWORD"]
    if os.getenv("TRIALMATCHAI_ES_CA_CERTS"):
        es_conf["ca_certs"] = os.environ["TRIALMATCHAI_ES_CA_CERTS"]
    if es_conf.get("ca_certs"):
        ca_path = Path(es_conf["ca_certs"]).expanduser()
        if not ca_path.is_absolute():
            es_conf["ca_certs"] = str((config_path.parent / ca_path).resolve())
    return cfg


def make_es_client(cfg: dict[str, Any]) -> Elasticsearch:
    es_conf = cfg["elasticsearch"]
    kwargs: dict[str, Any] = {
        "hosts": es_conf["hosts"],
        "basic_auth": (es_conf["username"], es_conf["password"]),
        "verify_certs": True,
        "request_timeout": es_conf.get("request_timeout", 60),
        "max_retries": es_conf.get("max_retries", 3),
        "retry_on_timeout": es_conf.get("retry_on_timeout", True),
    }
    if es_conf.get("ca_certs") and Path(es_conf["ca_certs"]).exists():
        kwargs["ca_certs"] = es_conf["ca_certs"]
    return Elasticsearch(**kwargs)
