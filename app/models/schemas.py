"""API data models for Hybrid RAG.

The response exposes both retrievers' contributions: each fused source reports its
RRF score, its raw BM25 and dense scores, and which retriever(s) found it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    source_id: str
    num_chunks: int
    embedding_provider: str
    vector_store: str


class HybridSource(BaseModel):
    rank: int
    chunk_id: str
    source_id: str
    chunk_index: int
    page: int | None = None
    fused_score: float = Field(..., description="Reciprocal Rank Fusion score.")
    bm25_score: float = Field(0.0, description="Raw BM25 score (0 if not retrieved lexically).")
    dense_score: float = Field(0.0, description="Raw cosine score (0 if not retrieved densely).")
    retrieved_by: list[str] = Field(
        default_factory=list, description="Retrievers that found this chunk: lexical/dense/fused."
    )
    text: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    per_retriever_k: int | None = Field(None, ge=1, le=50)
    source_filter: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[HybridSource]
    llm_provider: str
    fusion: str
    lexical_candidates: int
    dense_candidates: int
    unique_after_fusion: int
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    source_id: str
    document_id: str
    num_chunks: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_store: str
    llm_provider: str
    fusion: str
    bm25_k1: float
    bm25_b: float
    documents: int
    chunks: int
