"""
Structured logging with request-ID correlation.

Pure standard library (logging + contextvars) so it has zero third-party
dependencies and can be imported/tested anywhere in this repo.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid

from backend.config import LoggingConfig, load_config

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_configured = False


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str | None = None) -> str:
    rid = request_id or new_request_id()
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_var.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_PLAIN_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Idempotent: safe to call from multiple entry points (main.py,
    mcp_server.py, scripts) without producing duplicate handlers.
    """
    global _configured
    if _configured:
        return

    cfg = config or load_config().logging
    root = logging.getLogger()
    root.setLevel(cfg.level.upper())

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(_RequestIdFilter())
    if cfg.json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT))

    root.handlers = [handler]
    _configured = True
