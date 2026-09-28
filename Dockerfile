# syntax=docker/dockerfile:1.7
# Smart Resume Shortlisting System
#   docker build -t resume-shortlister .                       # app image (models baked in)
#   docker build --target test -t resume-shortlister:test .    # image + test suite

ARG PYTHON_IMAGE=python:3.12-slim-trixie

# ---------------------------------------------------------------- deps ----------------
# Third-party dependencies from uv.lock. On Linux, torch comes from the CPU-only index,
# which keeps the image several GB smaller than the default CUDA build.
FROM ${PYTHON_IMAGE} AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- models --------------
# Bake the embedding + reranker models into the image so containers run offline. Depends
# only on the lockfile and model names, so code changes never re-download ~1 GB.
FROM deps AS models
ARG PRELOAD_MODELS=1
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV HF_HOME=/opt/hf HF_HUB_DISABLE_TELEMETRY=1
RUN mkdir -p /opt/hf \
 && if [ "$PRELOAD_MODELS" = "1" ]; then /opt/venv/bin/python -c "\
import os; from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer(os.environ['EMBEDDING_MODEL']); CrossEncoder(os.environ['RERANKER_MODEL'])"; fi

# ---------------------------------------------------------------- builder -------------
FROM deps AS builder
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------- runtime -------------
FROM ${PYTHON_IMAGE} AS runtime
# Tesseract is the CPU fallback OCR; Chandra (GPU) runs as a separate service.
ARG TESSERACT_PACKAGES="eng hin mar"
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
      $(for lang in $TESSERACT_PACKAGES; do printf 'tesseract-ocr-%s ' "$lang"; done) \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
 && mkdir -p /data && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
COPY --from=models --chown=app:app /opt/hf /opt/hf
ARG PRELOAD_MODELS=1
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=${PRELOAD_MODELS} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL} \
    DATABASE_PATH=/data/shortlister.db
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "resume_shortlister.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- test ----------------
# Same OS packages (Tesseract) as production, plus dev dependencies and the test suite.
FROM runtime AS test
USER root
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests
RUN uv sync --frozen --no-editable && chown -R app:app /app
USER app
CMD ["pytest", "-q", "-p", "no:cacheprovider"]
