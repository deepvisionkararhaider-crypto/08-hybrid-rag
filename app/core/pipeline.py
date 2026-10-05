"""Hybrid RAG pipeline orchestrator.

Wires: loaders -> chunking -> (dense embeddings + BM25) -> :class:`HybridRetriever`
(BM25 + dense, fused with RRF) -> LLM generation, with citations and timings.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import load_document
from app.llm.provider import build_llm_provider
from app.models.schemas import DocumentInfo, HybridSource, IngestResponse, QueryResponse
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import build_vector_store

logger = get_logger(__name__)


class HybridRAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = build_embedder(
            self.settings.embedding_provider, self.settings.embedding_model_name
        )
        self.store = build_vector_store(self.settings.vector_store, self.embedder.dim)
        self.dense = Retriever(self.embedder, self.store)
        self.bm25 = BM25Index(k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        self.retriever = HybridRetriever(
            self.dense, self.bm25, fusion=self.settings.fusion, rrf_k=self.settings.rrf_k,
            dense_weight=self.settings.dense_weight, lexical_weight=self.settings.lexical_weight,
        )
        self.llm = build_llm_provider(self.settings)
        self._documents: dict[str, dict[str, Any]] = {}
        self._persist_path: Path = self.settings.data_path / "documents.json"
        self._load()
        logger.info(
            "HybridRAGPipeline ready | embedder=%s(dim=%d) store=%s lexical=bm25(k1=%.1f,b=%.2f) "
            "fusion=%s llm=%s | docs=%d chunks=%d",
            self.embedder.name, self.embedder.dim, self.store.name,
            self.settings.bm25_k1, self.settings.bm25_b, self.settings.fusion,
            self.llm.name, len(self._documents), self.store.count(),
        )

    def _index(self, text: str, source_id: str, metadata: dict, page_for_offset=None) -> tuple[str, int]:  # noqa: ANN001
        document_id = uuid.uuid4().hex
        chunks = chunk_text(
            text, chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap,
            base_metadata={"source_id": source_id, **metadata},
        )
        if not chunks:
            raise ValueError("Document produced no chunks after splitting.")
        records = []
        for ch in chunks:
            page = page_for_offset(ch.char_start) if page_for_offset else None
            records.append({
                "chunk_id": f"{document_id}::{ch.index}", "document_id": document_id,
                "source_id": source_id, "index": ch.index, "text": ch.text,
                "metadata": {"char_start": ch.char_start, "char_end": ch.char_end, "page": page, **metadata},
            })
        self.retriever.add(records)
        self._documents[source_id] = {
            "document_id": document_id, "source_id": source_id, "num_chunks": len(records),
            "metadata": metadata,
            "chunks": [{"chunk_id": r["chunk_id"], "index": r["index"], "text": r["text"], "metadata": r["metadata"]} for r in records],
        }
        self._persist()
        return document_id, len(records)

    def ingest_text(self, text: str, source_id: str | None = None, metadata: dict | None = None) -> IngestResponse:
        if not text or not text.strip():
            raise ValueError("Document text is empty; nothing to ingest.")
        source_id = source_id or f"doc-{uuid.uuid4().hex[:8]}"
        doc_id, n = self._index(text, source_id, dict(metadata or {}))
        return IngestResponse(document_id=doc_id, source_id=source_id, num_chunks=n,
                              embedding_provider=self.embedder.name, vector_store=self.store.name)

    def ingest_file(self, filename: str, data: bytes, source_id: str | None = None, metadata: dict | None = None) -> IngestResponse:
        loaded = load_document(filename, data)
        if not loaded.text.strip():
            raise ValueError(f"No text could be extracted from '{filename}'.")
        meta = dict(metadata or {})
        meta.setdefault("filename", filename)
        if loaded.page_spans:
            meta.setdefault("num_pages", len(loaded.page_spans))
        source_id = source_id or filename
        doc_id, n = self._index(loaded.text, source_id, meta, page_for_offset=loaded.page_for_offset)
        return IngestResponse(document_id=doc_id, source_id=source_id, num_chunks=n,
                              embedding_provider=self.embedder.name, vector_store=self.store.name)

    def query(self, question: str, top_k: int | None = None, per_retriever_k: int | None = None,
              source_filter: str | None = None) -> QueryResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        k = top_k or self.settings.top_k
        pk = per_retriever_k or self.settings.per_retriever_k
        t0 = time.perf_counter()
        hits, counts = self.retriever.search(question, top_k=k, per_retriever_k=pk, source_filter=source_filter)
        t1 = time.perf_counter()
        sources = [
            HybridSource(rank=i, chunk_id=h.chunk_id, source_id=h.source_id, chunk_index=h.chunk_index,
                         page=h.page, fused_score=round(h.fused_score, 6), bm25_score=round(h.bm25_score, 6),
                         dense_score=round(h.dense_score, 6),
                         retrieved_by=sorted(h.retrieved_by), text=h.text)
            for i, h in enumerate(hits, start=1)
        ]
        answer = self.llm.generate(question, [h.text for h in hits])
        t2 = time.perf_counter()
        return QueryResponse(
            question=question, answer=answer, sources=sources, llm_provider=self.llm.name,
            fusion=self.settings.fusion, lexical_candidates=counts["lexical"], dense_candidates=counts["dense"],
            unique_after_fusion=len(sources),
            timings_ms={"retrieval": round((t1 - t0) * 1000, 2), "generation": round((t2 - t1) * 1000, 2), "total": round((t2 - t0) * 1000, 2)},
        )

    def list_documents(self) -> list[DocumentInfo]:
        return [DocumentInfo(source_id=d["source_id"], document_id=d["document_id"],
                             num_chunks=d["num_chunks"], metadata=d.get("metadata", {}))
                for d in self._documents.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._documents), "chunks": self.store.count(),
            "embedding_provider": self.embedder.name, "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim, "vector_store": self.store.name, "llm_provider": self.llm.name,
            "fusion": self.settings.fusion, "bm25_k1": self.settings.bm25_k1, "bm25_b": self.settings.bm25_b,
        }

    def reset(self) -> None:
        self._documents.clear(); self.retriever.clear(); self._persist()

    def _persist(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps({"documents": self._documents}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist documents: %s", exc)

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for source_id, doc in raw.get("documents", {}).items():
                records = [{"chunk_id": c["chunk_id"], "document_id": doc["document_id"], "source_id": source_id,
                            "index": c["index"], "text": c["text"], "metadata": c.get("metadata", {})}
                           for c in doc.get("chunks", [])]
                if records:
                    self.retriever.add(records)
                self._documents[source_id] = doc
            logger.info("Restored %d document(s) from disk.", len(self._documents))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load persisted documents: %s", exc)


_pipeline: HybridRAGPipeline | None = None


def get_pipeline() -> HybridRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridRAGPipeline(get_settings())
    return _pipeline
