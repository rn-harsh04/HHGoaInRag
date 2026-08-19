"""Application configuration from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent


def get_data_dir() -> Path:
    if (BACKEND_DIR / "data").exists():
        return BACKEND_DIR / "data"
    if (ROOT_DIR / "data").exists():
        return ROOT_DIR / "data"
    return BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[str(ROOT_DIR / ".env"), str(BACKEND_DIR / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    gemini_api_key: str = ""
    hf_token: str = ""

    chroma_persist_dir: Path = Field(default_factory=lambda: get_data_dir() / "chroma")
    faiss_index_path: Path = Field(default_factory=lambda: get_data_dir() / "faiss" / "hnsw.index")
    faiss_id_map_path: Path = Field(default_factory=lambda: get_data_dir() / "faiss" / "id_map.json")
    chunk_metadata_path: Path = Field(default_factory=lambda: get_data_dir() / "processed" / "chunks.parquet")
    parents_path: Path = Field(default_factory=lambda: get_data_dir() / "processed" / "parents.jsonl")
    bm25_index_dir: Path = Field(default_factory=lambda: get_data_dir() / "bm25")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_model: str = "gemini-3.1-flash-lite"

    retrieval_min_score: float = 0.42
    groundedness_min_score: float = 0.45
    retrieval_budget_ms: int = 200

    top_k_per_channel: int = 15
    final_top_k: int = 6
    rrf_k: int = 60
    max_parents: int = 4

    cors_origins: str = "*"
    log_refusals: bool = True
    refusal_log_path: Path = Field(default_factory=lambda: get_data_dir() / "logs" / "refusals.jsonl")
    benchmark_api_key: str = ""

    warmup_queries: int = 5
    stt_timeout_sec: float = 15.0
    stt_max_retries: int = 2
    max_audio_duration_sec: float = 28.0
    max_audio_bytes: int = 10 * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
