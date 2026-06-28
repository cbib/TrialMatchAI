from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRecord:
    nct_id: str
    source_url: str
    source_hash: str
    fetched_at: str
    last_update_posted: str | None
    processing_status: str
    error_summary: str | None = None


class RegistryManifest:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_latest(self) -> dict[str, ManifestRecord]:
        latest: dict[str, ManifestRecord] = {}
        if not self.path.exists():
            return latest
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    record = ManifestRecord(**raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                latest[record.nct_id] = record
        return latest

    def append(self, record: ManifestRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def source_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
