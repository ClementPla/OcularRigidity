"""Logger setup shared by the cohort scripts."""

import logging
from pathlib import Path


def setup_logging(name: str, log_file: Path, rank_tag: str = "") -> logging.Logger:
    """Logger writing to both ``log_file`` and stderr.

    ``rank_tag`` is interpolated into the format string for sharded stages
    (``" [shard 0]"``); the default reproduces the unsharded format exactly.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if re-run in a notebook

    fmt = logging.Formatter(f"%(asctime)s | %(levelname)s |{rank_tag} %(message)s")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
