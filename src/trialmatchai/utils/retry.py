from __future__ import annotations

import random
import time
from typing import Callable, Optional, Tuple, TypeVar

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: float = 0.1,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    logger: Optional[object] = None,
    action: str = "operation",
) -> T:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if logger is not None:
                logger.warning(
                    "%s failed (attempt %s/%s): %s", action, attempt, retries, exc
                )
            if attempt < retries:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, jitter)
                time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{action} failed after {retries} attempts")
