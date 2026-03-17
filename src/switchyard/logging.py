"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging and structlog for local development."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
    ]
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout, force=True)
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger."""

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_request_context(**values: Any) -> None:
    """Bind request-scoped values into structlog contextvars."""

    structlog.contextvars.bind_contextvars(**values)


def clear_request_context() -> None:
    """Clear any bound request-scoped contextvars."""

    structlog.contextvars.clear_contextvars()
