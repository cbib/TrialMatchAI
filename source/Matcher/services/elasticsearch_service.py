from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from elasticsearch import Elasticsearch

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def build_elasticsearch_client(config: Dict[str, Any]) -> Elasticsearch:
    es_cfg = config["elasticsearch"]
    paths = config.get("paths", {})
    kwargs: dict[str, Any] = {
        "hosts": [es_cfg["host"]],
        "basic_auth": (es_cfg["username"], es_cfg["password"]),
        "request_timeout": es_cfg["request_timeout"],
        "retry_on_timeout": es_cfg["retry_on_timeout"],
    }
    ca_certs = paths.get("docker_certs")
    if ca_certs and Path(ca_certs).exists():
        kwargs["ca_certs"] = ca_certs
    return Elasticsearch(**kwargs)


def ensure_elasticsearch(es_client: Any, config: Dict[str, Any]) -> bool:
    if _ping(es_client):
        return True

    es_cfg = config.get("elasticsearch", {})
    auto_start = bool(es_cfg.get("auto_start", False))
    if not auto_start:
        logger.error(
            "Elasticsearch is not reachable at %s and auto_start is disabled.",
            es_cfg.get("host"),
        )
        return False

    script_path = es_cfg.get("start_script", "elasticsearch/apptainer-run-es.sh")
    timeout = int(es_cfg.get("start_timeout", 600))
    script = _resolve_repo_path(script_path)

    if not script.exists():
        logger.error("Elasticsearch start script not found: %s", script)
        return False

    if shutil.which("apptainer") is None:
        logger.error("Apptainer is not available in PATH.")
        return False

    logger.info("Starting Elasticsearch via Apptainer script: %s", script)
    try:
        subprocess.run(
            ["bash", str(script)],
            cwd=str(script.parent),
            check=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        logger.error("Elasticsearch start timed out after %s seconds.", timeout)
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("Elasticsearch start failed: %s", exc)
        return False

    return _wait_for_es(es_client, timeout=60)


def _ping(es_client: Any) -> bool:
    try:
        return bool(es_client.ping())
    except Exception:
        return False


def _wait_for_es(es_client: Any, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _ping(es_client):
            return True
        time.sleep(5)
    logger.error("Elasticsearch did not become available within %s seconds.", timeout)
    return False


def _resolve_repo_path(relative_path: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return (root / relative_path).resolve()
