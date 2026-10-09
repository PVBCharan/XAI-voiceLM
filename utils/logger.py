"""Application logging utility with colored console output and optional file logging."""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "assistant", log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configures and returns a logger instance with formatted console and optional file handlers."""
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if requested)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# Default root logger
logger = setup_logger()
