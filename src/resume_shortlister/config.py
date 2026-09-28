"""Runtime configuration from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from resume_shortlister.application.dto import UploadLimits
from resume_shortlister.domain.scoring import ScoringPolicy

MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server -------------------------------------------------------------------
    log_level: str = "INFO"
    database_path: Path = Path("data/shortlister.db")
    max_concurrent_runs: int = Field(1, ge=1, le=8)
    warmup_models: bool = False  # load ML models at startup instead of on first use

    # --- Uploads ------------------------------------------------------------------
    max_files: int = Field(50, ge=1, le=500)
    max_file_mb: float = Field(10, gt=0)
    max_total_mb: float = Field(100, gt=0)
    max_pdf_pages: int = Field(10, ge=1, le=100)

    # --- OCR ----------------------------------------------------------------------
    # auto: Chandra when its server is reachable, otherwise Tesseract.
    ocr_engine: Literal["auto", "chandra", "tesseract", "none"] = "auto"
    chandra_api_base: str = "http://localhost:8001/v1"
    chandra_model_name: str = "chandra"
    chandra_api_key: SecretStr = SecretStr("EMPTY")
    chandra_prompt_type: Literal["ocr_layout", "ocr"] = "ocr_layout"
    chandra_max_retries: int = Field(2, ge=0, le=10)
    tesseract_langs: str = "eng"  # e.g. "eng+hin+mar" (install the matching language packs)
    pdf_min_chars_per_page: int = Field(40, ge=0)
    ocr_dpi: int = Field(200, ge=72, le=600)

    # --- Models -------------------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_query_prompt: str = "query: "
    embedding_document_prompt: str = "passage: "
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    model_device: str = "cpu"  # cpu | cuda | mps | auto

    # --- Scoring ------------------------------------------------------------------
    retrieval_top_k: int = Field(20, ge=1, le=200)
    default_shortlist_size: int = Field(10, ge=1, le=50)
    semantic_weight: float = Field(0.3, ge=0)
    coverage_weight: float = Field(0.7, ge=0)
    semantic_floor: float = 0.78
    semantic_ceiling: float = 0.93
    max_requirements: int = Field(15, ge=1, le=50)
    chunk_words: int = Field(50, ge=20, le=400)
    chunk_overlap_words: int = Field(15, ge=0)
    evidence_chunks_per_requirement: int = Field(2, ge=1, le=10)
    social_max_bonus: float = Field(10.0, ge=0, le=50)

    # --- AI summaries (any OpenAI-compatible chat API; default GPT-OSS 120B on Groq) --
    # Disabled unless LLM_API_KEY is set. Summaries explain a ranking, never change it.
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_display_name: str = "GPT-OSS 120B [Groq]"
    # Per-request output cap. Keep it small: providers count it against tokens-per-minute
    # limits (Groq's free tier allows 8,000/min, so the model's 65,536 maximum would fail).
    llm_max_tokens: int = Field(1200, ge=128, le=65536)
    llm_reasoning_effort: str = "low"  # for reasoning models; set "" for other models
    llm_temperature: float = Field(0.3, ge=0, le=2)
    llm_timeout_seconds: float = Field(60, gt=0)
    llm_max_candidates: int = Field(5, ge=1, le=50)  # summaries per run (top shortlisted)
    llm_max_resume_chars: int = Field(4000, ge=500, le=50_000)

    # --- Public profiles ------------------------------------------------------------
    github_token: SecretStr | None = None
    social_timeout_seconds: float = Field(8.0, gt=0)
    social_cache_ttl_seconds: float = Field(3600, ge=0)

    def scoring_policy(self) -> ScoringPolicy:
        return ScoringPolicy(
            retrieval_top_k=self.retrieval_top_k,
            shortlist_size=self.default_shortlist_size,
            semantic_weight=self.semantic_weight,
            coverage_weight=self.coverage_weight,
            semantic_floor=self.semantic_floor,
            semantic_ceiling=self.semantic_ceiling,
            social_max_bonus=self.social_max_bonus,
            max_requirements=self.max_requirements,
            chunk_words=self.chunk_words,
            chunk_overlap_words=self.chunk_overlap_words,
            evidence_chunks_per_requirement=self.evidence_chunks_per_requirement,
        )

    def upload_limits(self) -> UploadLimits:
        return UploadLimits(
            max_files=self.max_files,
            max_file_bytes=int(self.max_file_mb * MB),
            max_total_bytes=int(self.max_total_mb * MB),
        )
