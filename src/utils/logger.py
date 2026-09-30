import logging
import sys
from pathlib import Path
from datetime import datetime
import coloredlogs


def setup_logger(
    name: str = 'fake_news_detection',
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with colored console output and file logging.

    Args:
        name:     Logger name
        log_file: Path to log file (full training log)
        level:    Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ── Console handler (colored) ──────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    coloredlogs.install(
        level  = level,
        logger = logger,
        fmt    = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ── File handler (full log) ────────────────────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_metrics_logger(metrics_log_file: str) -> logging.Logger:
    """
    Create a dedicated logger that writes epoch metrics to a clean text file.

    One line per epoch — easy to grep, plot, or share.
    No timestamps or log levels in the output, just the metrics.

    Args:
        metrics_log_file: Path to the metrics log file (e.g. logs/exp/metrics.log)

    Returns:
        Configured metrics logger
    """
    name = 'metrics'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False   # don't bubble up to root logger

    Path(metrics_log_file).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(metrics_log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))  # plain text only
    logger.addHandler(handler)

    return logger