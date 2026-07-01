"""Stage-level skip/resume for the build pipeline via fingerprinted completion state.

Synthesizes the time-tested incremental-build patterns:

- FINGERPRINTS over mtime alone. Bazel detects changes by hashing a step's *inputs and
  command*, not timestamps, because mtime is unsound: it decreases when you check out an
  older revision, and it is blind to changes in the command/config. We fingerprint each
  stage's INPUTS (a fast path+size+mtime digest, as Nextflow's ``-resume`` does per file)
  PLUS a config digest PLUS a stage code-version, so a stage re-runs when its inputs,
  config, OR logic change -- not just when a file's clock moves.
- COMPLETION SENTINEL + dependency chaining (Hadoop/Spark ``_SUCCESS``; dbt state:modified).
  A stage records its output fingerprint; the next stage stores the upstream fingerprint it
  consumed and skips instantly when the upstream, config, and code are unchanged and its own
  output is still present -- no per-record walk.
- ATOMIC writes (Spark commit protocol). The manifest is written temp-then-os.replace and
  only AFTER a stage fully succeeds, so a crash never leaves a false "complete" marker.
- A ``force`` escape hatch always rebuilds (make clean / dbt --full-refresh).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def dir_fingerprint(
    path: str | Path, *, pattern: str = "*.json", include_dirs: bool = False
) -> str:
    """Fast fingerprint of a directory listing from each entry's (name, size, mtime_ns).

    No file contents are read (Nextflow-style path+size+mtime), so this stays cheap even
    for large corpora. Detects added, removed, and modified entries. With ``include_dirs``
    it fingerprints immediate sub-directories by (name, mtime_ns) -- useful for a
    per-trial-directory corpus. Returns "" for a missing/empty directory.
    """
    root = Path(path)
    if not root.exists():
        return ""
    items: list[str] = []
    entries = root.iterdir() if include_dirs else root.glob(pattern)
    for entry in sorted(entries, key=lambda p: p.name):
        try:
            st = entry.stat()
        except OSError:
            continue
        if include_dirs:
            if entry.is_dir():
                items.append(f"{entry.name}/:{st.st_mtime_ns}")
        else:
            items.append(f"{entry.name}:{st.st_size}:{st.st_mtime_ns}")
    if not items:
        return ""
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def digest(*parts: Any) -> str:
    """Stable sha256 over JSON-serialized parts (composing input/config/version signals)."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Write JSON atomically: temp file in the same dir, fsync, then os.replace.

    os.replace is atomic on POSIX and Windows for a same-filesystem rename, so a reader
    never sees a half-written manifest and a crash cannot corrupt the existing one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def stage_is_current(
    entry: dict[str, Any] | None, *, fingerprint: str, output_present: bool
) -> bool:
    """True iff a recorded stage can be skipped: it completed, its fingerprint matches the
    freshly computed one (non-empty), and its output is still present.

    A missing entry, a "complete"-less/failed entry, a changed fingerprint (inputs, config,
    or code version differ), or a vanished output all force the stage to re-run.
    """
    if not entry:
        return False
    return (
        entry.get("status") == "complete"
        and bool(fingerprint)
        and entry.get("fingerprint") == fingerprint
        and output_present
    )
