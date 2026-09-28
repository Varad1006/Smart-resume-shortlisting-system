"""Bi-encoder adapter (sentence-transformers). Default: intfloat/multilingual-e5-small."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from resume_shortlister.infrastructure.nlp.device import resolve_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Lazily loaded, thread-safe embedder returning L2-normalised float32 vectors.

    E5 models expect "query: " / "passage: " prefixes; they are configurable so other
    models (which need no prefix) work too.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        query_prompt: str = "",
        document_prompt: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        self._load()

    def _load(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                device = resolve_device(self._device)
                logger.info("Loading embedding model %s on %s", self.model_name, device)
                self._model = SentenceTransformer(self.model_name, device=device)
            return self._model

    def _encode(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        model = self._load()
        if not texts:
            # Renamed in sentence-transformers 6; keep working on 5.x.
            get_dimension = (
                getattr(model, "get_embedding_dimension", None)
                or model.get_sentence_embedding_dimension
            )
            return np.zeros((0, get_dimension() or 0), dtype=np.float32)
        vectors = model.encode(
            list(texts),
            prompt=prompt or None,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._query_prompt)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self._document_prompt)
