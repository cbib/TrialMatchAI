import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def set_request_id(request_id: str) -> contextvars.Token:
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure logging for the application."""
    level = os.getenv("TRIALMATCHAI_LOG_LEVEL", "INFO").upper()
    use_json = os.getenv("TRIALMATCHAI_LOG_JSON", "0") in {"1", "true", "TRUE"}
    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logging.basicConfig(level=level, handlers=[handler])
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger