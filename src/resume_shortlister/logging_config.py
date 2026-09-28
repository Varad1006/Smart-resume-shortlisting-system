"""Logging setup. Resume content and candidate names are never logged, only ids/counts."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
