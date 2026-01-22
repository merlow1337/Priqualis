"""
Application configuration using pydantic-settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    priqualis_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Paths
    data_raw_path: Path = Path("./data/raw")
    data_processed_path: Path = Path("./data/processed")
    rules_config_path: Path = Path("./config/rules")

    # DuckDB
    duckdb_path: Path = Path("./data/priqualis.duckdb")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "claims_embeddings"
    qdrant_api_key: str | None = None

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 32
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Search
    bm25_k1: float = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    search_top_k: int = Field(default=50, ge=1, le=500)
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Streamlit
    streamlit_port: int = 8501

    # Scoring Weights (used by ImpactScorer)
    scoring_error_rejection_risk: float = Field(default=0.9, ge=0.0, le=1.0)
    scoring_warning_rejection_risk: float = Field(default=0.3, ge=0.0, le=1.0)
    scoring_autofix_cost: float = Field(default=0.2, ge=0.0)
    scoring_manual_fix_cost: float = Field(default=1.0, ge=0.0)
    scoring_tariff_baseline: float = Field(default=5000.0, ge=0.0)

    # Hybrid Search Parameters
    rrf_k: int = Field(default=60, ge=1, description="RRF constant")
    hybrid_bm25_candidates: int = Field(default=200, ge=1)
    hybrid_vector_candidates: int = Field(default=50, ge=1)

    # Vector Store (HNSW)
    hnsw_m: int = Field(default=16, ge=4, le=64)
    hnsw_ef_construct: int = Field(default=100, ge=10)
    vector_upsert_batch_size: int = Field(default=100, ge=1)
    vector_default_dimension: int = Field(default=384, ge=1)

    # LLM Explainer
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = Field(default=300, ge=1, le=4096)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_language: Literal["pl", "en"] = "pl"

    # FPA Tracking
    fpa_default_period_days: int = Field(default=30, ge=1)
    fpa_trend_window: int = Field(default=7, ge=1)
    fpa_top_reasons_limit: int = Field(default=5, ge=1)

    # AutoFix
    autofix_safe_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    autofix_base_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # API Metadata
    api_version: str = "0.1.0"
    api_title: str = "Priqualis API"
    cors_allow_origins: list[str] = ["*"]

    @property
    def is_production(self) -> bool:
        return self.priqualis_env == "production"

    @property
    def is_development(self) -> bool:
        return self.priqualis_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()