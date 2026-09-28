"""Download (and smoke-test) the configured models into the Hugging Face cache.

Used by the Docker build so the image runs fully offline:
    python -m resume_shortlister.download_models
"""

from __future__ import annotations

import logging

from resume_shortlister.config import Settings
from resume_shortlister.infrastructure.nlp.embedder import SentenceTransformerEmbedder
from resume_shortlister.infrastructure.nlp.reranker import CrossEncoderReranker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prompt=settings.embedding_query_prompt,
        document_prompt=settings.embedding_document_prompt,
    )
    vectors = embedder.embed_documents(["Python developer with FastAPI experience"])
    reranker = CrossEncoderReranker(settings.reranker_model)
    scores = reranker.score([("Python developer", "Five years of Python and FastAPI.")])
    logging.info(
        "Models ready: %s (dim %d), %s (sample score %.3f)",
        settings.embedding_model,
        vectors.shape[1],
        settings.reranker_model,
        scores[0],
    )


if __name__ == "__main__":
    main()
