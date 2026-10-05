"""Retrieval: embed a query and return the most similar stored chunks.

A thin, honest wrapper over the embedder + vector store. Chunk records are stored
as the vector-store payload so a search result carries everything needed to build
a citation (id, source, index, text, page) without a second lookup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def add(self, records: list[dict[str, Any]]) -> None:
        """Index chunk records. Each record must contain ``chunk_id`` and ``text``."""
        if not records:
            return
        texts = [r["text"] for r in records]
        ids = [r["chunk_id"] for r in records]
        vectors = self.embedder.embed(texts)
        self.store.add(ids, vectors, records)

    def search(
        self, query: str, k: int, source_filter: str | None = None
    ) -> list[SearchResult]:
        """Return up to ``k`` results ranked by cosine similarity (highest first)."""
        if k <= 0 or not query.strip():
            return []
        query_vector = self.embedder.embed_one(query)
        predicate = None
        if source_filter:
            predicate = lambda payload: payload.get("source_id") == source_filter  # noqa: E731
        return self.store.search(query_vector, k, predicate)

    def search_vector(
        self, query_vector: np.ndarray, k: int, source_filter: str | None = None
    ) -> list[SearchResult]:
        """Same as :meth:`search` but reuses a precomputed query embedding."""
        if k <= 0:
            return []
        predicate = None
        if source_filter:
            predicate = lambda payload: payload.get("source_id") == source_filter  # noqa: E731
        return self.store.search(query_vector, k, predicate)
