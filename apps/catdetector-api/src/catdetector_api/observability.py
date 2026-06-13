import json
import logging
import logging.config
import os
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

REQUEST_ID = ContextVar("request_id", default="-")

_STANDARD_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", REQUEST_ID.get()),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_KEYS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def api_log_level() -> str:
    return os.environ.get("CATDETECTOR_LOG_LEVEL", "INFO").upper()


def api_logging_dict_config() -> dict[str, Any]:
    level = api_log_level()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": "catdetector_api.observability.JsonLogFormatter"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "_granian": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "granian.access": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "catdetector_api": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }


def configure_api_logging() -> None:
    logging.config.dictConfig(api_logging_dict_config())


@contextmanager
def bind_request_id(request_id: str) -> Iterator[None]:
    token = REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        REQUEST_ID.reset(token)
