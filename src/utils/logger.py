"""
Logging utility using Loguru
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def setup_logger() -> None:
    """Configure logger with file and console handlers"""

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.logging.level,
    )

    # Create log directory if it doesn't exist
    settings.logging.log_dir.mkdir(parents=True, exist_ok=True)

    # File handler
    logger.add(
        settings.logging.log_dir / "delivery_ml_{time:YYYY-MM-DD}.log",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.logging.level,
    )


def get_logger(name: str):
    """Get a logger instance with the given name"""
    return logger.bind(name=name)


# Setup logger on module import
setup_logger()
