"""Shared pytest fixtures for Hybrid RAG: isolated, deterministic pipeline."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("VECTOR_STORE", "numpy")
os.environ.setdefault("LLM_PROVIDER", "extractive")

from app.core.config import Settings  # noqa: E402
from app.core.pipeline import HybridRAGPipeline  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_provider="hashing", vector_store="numpy", llm_provider="extractive",
        chunk_size=300, chunk_overlap=40, top_k=5, per_retriever_k=10, fusion="rrf", rrf_k=60,
        data_dir=str(tmp_path / "data"),
    )


@pytest.fixture()
def pipeline(settings) -> HybridRAGPipeline:
    return HybridRAGPipeline(settings)


@pytest.fixture()
def sample_docs() -> list[dict]:
    """Corpus containing BOTH lexical-only and semantic-only retrieval cases:
    - an exact identifier ("ERR-4032") that dense embeddings can blur but BM25 nails;
    - a paraphrase case where the document says "spend" and the query says "costs".
    """
    return [
        {
            "source_id": "error_codes.md",
            "text": (
                "# Error Codes\n\n"
                "ERR-4032 indicates the authentication token has expired. Re-issue the "
                "token and retry the request. This error is returned by the gateway "
                "before the request reaches any backend service.\n\n"
                "ERR-5001 indicates a downstream timeout. Check the dependency health "
                "dashboard and consider increasing the client timeout."
            ),
        },
        {
            "source_id": "cost_guide.md",
            "text": (
                "# Reducing Cloud Spend\n\n"
                "Organisations lower their cloud expenditure by right-sizing instances, "
                "purchasing commitment discounts and deleting orphaned volumes. Moving "
                "cold data to archival storage tiers also trims the monthly bill "
                "substantially."
            ),
        },
        {
            "source_id": "scaling.md",
            "text": (
                "# Autoscaling\n\n"
                "Elastic capacity follows demand so you never pay for idle servers. "
                "Horizontal autoscaling adds instances under load and removes them when "
                "traffic subsides."
            ),
        },
    ]
