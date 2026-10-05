"""Application configuration (pydantic-settings).

Hybrid RAG runs two complementary retrievers -- lexical **BM25** and dense
**semantic** -- and fuses their rankings with **Reciprocal Rank Fusion**. The BM25
parameters and the fusion constant are configurable here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Embeddings / dense vector store ---
    embedding_provider: str = "auto"  # auto | hashing | sentence-transformers
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_store: str = "auto"  # auto | numpy | faiss

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- BM25 (lexical retriever) ---
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- Hybrid retrieval ---
    top_k: int = 5  # final fused context chunks sent to the LLM
    per_retriever_k: int = 10  # candidates fetched from EACH retriever
    fusion: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    dense_weight: float = 0.5  # used only when fusion == "weighted"
    lexical_weight: float = 0.5  # used only when fusion == "weighted"

    # --- Generation ---
    llm_provider: str = "extractive"  # extractive | openai | huggingface
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    huggingface_api_key: str = ""
    huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    data_dir: str = "./data"
    index_dir: str = "./index"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
