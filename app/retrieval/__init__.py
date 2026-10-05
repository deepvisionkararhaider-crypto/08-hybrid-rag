"""Retrieval package: shared backends + BM25 + hybrid fusion."""

from app.retrieval.bm25 import BM25Index, LexicalHit
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.hybrid import HybridHit, HybridRetriever
from app.retrieval.retriever import Retriever
from app.retrieval.rrf import ranked_entries, reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult, VectorStore, build_vector_store

__all__ = [
    "BM25Index", "LexicalHit", "Embedder", "build_embedder", "HybridHit",
    "HybridRetriever", "Retriever", "ranked_entries", "reciprocal_rank_fusion",
    "SearchResult", "VectorStore", "build_vector_store",
]
