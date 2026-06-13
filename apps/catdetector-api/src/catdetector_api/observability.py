import json
import logging
import logging.config
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from catdetector_api.settings import ApiSettings, LogFormat

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


def log_payload(record: logging.LogRecord) -> dict[str, Any]:
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

    return payload


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = log_payload(record)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = log_payload(record)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in payload.items()
        )


def api_logging_dict_config(settings: ApiSettings) -> dict[str, Any]:
    formatter = (
        "catdetector_api.observability.JsonLogFormatter"
        if settings.log_format == LogFormat.JSON
        else "catdetector_api.observability.TextLogFormatter"
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"structured": {"()": formatter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "_granian": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
            "granian.access": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
            "catdetector_api": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
        },
    }


def configure_api_logging(settings: ApiSettings) -> None:
    logging.config.dictConfig(api_logging_dict_config(settings))


@contextmanager
def bind_request_id(request_id: str) -> Iterator[None]:
    token = REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        REQUEST_ID.reset(token)
