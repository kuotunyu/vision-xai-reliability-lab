"""Central logging setup, called once by the CLI entry point."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | None = None, log_file: Path | None = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Level resolution order: explicit argument, ``VISION_XAI_LOG_LEVEL`` env var, INFO.
    """
    resolved = level or os.environ.get("VISION_XAI_LOG_LEVEL") or "INFO"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=resolved.upper(),
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
