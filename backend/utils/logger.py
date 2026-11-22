"""Structured logger for SmartDocs backend.

Provides a unified console logging configuration across all modules.
"""

import logging
import sys


def get_logger(name: str = "smartdocs") -> logging.Logger:
    """Create and return a configured logger instance.

    Args:
        name: Name of the logger or module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
