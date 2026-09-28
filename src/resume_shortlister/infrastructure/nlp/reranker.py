"""Cross-encoder adapter. Default: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from resume_shortlister.infrastructure.nlp.device import resolve_device

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.exp(-np.logaddexp(0.0, -logits))


class CrossEncoderReranker:
    """Scores (query, passage) pairs as probabilities in [0, 1].

    MS MARCO cross-encoders emit raw logits (roughly -12..+12). The original project
    multiplied those by 100 and displayed them with ``% 100``; here they go through a
    sigmoid so they are genuine, comparable probabilities.
    """

    def __init__(
        self, model_name: str, *, device: str = "cpu", batch_size: int = 32, max_length: int = 512
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        self._load()

    def _load(self) -> CrossEncoder:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                device = resolve_device(self._device)
                logger.info("Loading reranker model %s on %s", self.model_name, device)
                self._model = CrossEncoder(
                    self.model_name, device=device, max_length=self._max_length
                )
            return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        import torch

        logits = self._load().predict(
            [list(pair) for pair in pairs],
            batch_size=self._batch_size,
            activation_fn=torch.nn.Identity(),  # always raw logits, whatever the model config
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return sigmoid(np.asarray(logits, dtype=np.float64).reshape(-1)).tolist()
