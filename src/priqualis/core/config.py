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