"""Structured logging: JSON to logs/app.jsonl, pretty console in dev."""

from __future__ import annotations

import logging
import sys

import structlog

from src.config import get_settings

_configured = False


def configure_logging() -> None:
    """Idempotent. Sets up stdlib logging + structlog to write JSON to
    logs/app.jsonl and pretty-print to the console."""
    global _configured
    if _configured:
        return

    cfg = get_settings()
    log_dir = cfg.project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.jsonl"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [file_handler, console_handler]

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    )
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
        )
    )

    _configured = True


def get_logger(name: str = __name__):
    configure_logging()
    return structlog.get_logger(name)
