"""Loguru logger configuration for PyroSense AI.

All modules should import `logger` from here to keep formatting consistent.

Example:
    >>> from utils.logger import logger
    >>> logger.info("PyroSense AI started")
"""

from __future__ import annotations

import os
import sys
from loguru import logger as _logger


def _configure_logger() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        enqueue=True,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


_configure_logger()

logger = _logger

