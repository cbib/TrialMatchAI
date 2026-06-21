from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def log_timing(logger, label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", label, elapsed)
