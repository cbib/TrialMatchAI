"""Stage-level skip/resume for the build pipeline via fingerprinted completion state.

A stage re-runs when its inputs (path+size+mtime digest), config, or code version change; mtime
alone is unsound across checkouts. Each stage records its output and upstream fingerprints so a
downstream stage skips instantly (no per-record walk) when upstream, config, and code are
unchanged and its output is present. The manifest is written atomically only after success, so a
crash never leaves a false "complete" marker; ``force`` always rebuilds.
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
    """Fast fingerprint of a directory from each entry's (name, size, mtime_ns); no contents read.

    Detects added/removed/modified entries. ``include_dirs`` folds each sub-directory's
    contained files into the signature. Returns "" for a missing/empty directory.
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
                # LanceDB (copy-on-write) rewrites nested files without touching the dir mtime,
                # so fold contained files' size+mtime into the signature.
                inner: list[str] = []
                for child in sorted(entry.rglob("*"), key=lambda p: str(p)):
                    try:
                        cst = child.stat()
                    except OSError:
                        continue
                    if child.is_file():
                        rel = child.relative_to(entry)
                        inner.append(f"{rel}:{cst.st_size}:{cst.st_mtime_ns}")
                inner_digest = hashlib.sha256("\n".join(inner).encode("utf-8")).hexdigest()
                items.append(f"{entry.name}/:{st.st_mtime_ns}:{inner_digest}")
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
    """Write JSON atomically (temp + fsync + os.replace) so a reader never sees a half-written file."""
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
    """True iff a stage can be skipped: completed, non-empty fingerprint matches, output present."""
    if not entry:
        return False
    return (
        entry.get("status") == "complete"
        and bool(fingerprint)
        and entry.get("fingerprint") == fingerprint
        and output_present
    )
