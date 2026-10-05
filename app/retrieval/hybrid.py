"""Hybrid retrieval -- lexical BM25 + dense semantic, fused with RRF.

The core of Hybrid RAG. Two retrievers with complementary failure modes run in
parallel over the same chunk set:

* **BM25 (lexical)** excels at exact terms — identifiers, acronyms, error codes,
  product names — where embeddings blur or miss.
* **Dense (semantic)** excels at paraphrase and synonymy — where the query and the
  document share meaning but not words.

Their ranked lists are merged with **Reciprocal Rank Fusion** (`rrf`) — robust
because it uses only rank, so BM25's unbounded scores and cosine's [-1,1] scores
never need normalisation. A `weighted` fusion (min-max normalised, then a weighted
sum) is offered as an alternative and is clearly documented as such.

A chunk found by *both* retrievers is promoted by RRF, which is exactly the intent:
agreement between lexical and semantic evidence is a strong relevance signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.bm25 import BM25Index, LexicalHit
from app.retrieval.retriever import Retriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult


@dataclass
class HybridHit:
    chunk_id: str
    source_id: str
    chunk_index: int
    page: int | None
    text: str
    fused_score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    retrieved_by: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        bm25: BM25Index,
        *,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    # ---- indexing -----------------------------------------------------------
    def add(self, records: list[dict]) -> None:
        """Index the same records into both retrievers (keeps them in lockstep)."""
        if not records:
            return
        self.dense.add(records)
        self.bm25.add(
            [r["chunk_id"] for r in records],
            [r["text"] for r in records],
            records,
        )

    def count(self) -> int:
        return self.dense.store.count()

    def clear(self) -> None:
        self.dense.store.clear()
        self.bm25.clear()

    # ---- search -------------------------------------------------------------
    def search(
        self,
        question: str,
        top_k: int = 5,
        per_retriever_k: int = 10,
        source_filter: str | None = None,
    ) -> tuple[list[HybridHit], dict[str, int]]:
        lean = lambda p: (p.get("source_id") == source_filter)  # noqa: E731
        dense_hits = self.dense.search(question, per_retriever_k, source_filter=source_filter)
        lexical_hits: list[LexicalHit] = self.bm25.search(question, per_retriever_k)
        if source_filter:
            lexical_hits = [h for h in lexical_hits if lean(h.payload)]

        counts = {"lexical": len(lexical_hits), "dense": len(dense_hits)}

        if self.fusion == "weighted":
            hits = self._weighted_fuse(dense_hits, lexical_hits, top_k)
        else:
            hits = self._rrf_fuse(dense_hits, lexical_hits, top_k)
        return hits, counts

    def _base_hit(self, payload: dict, ident: str) -> HybridHit:
        meta = payload.get("metadata", {}) or {}
        return HybridHit(
            chunk_id=payload.get("chunk_id", ident),
            source_id=payload.get("source_id", ""),
            chunk_index=int(payload.get("index", 0)),
            page=meta.get("page"),
            text=payload.get("text", ""),
            fused_score=0.0,
            payload=payload,
        )

    def _rrf_fuse(
        self, dense_hits: list[SearchResult], lexical_hits: list[LexicalHit], top_k: int
    ) -> list[HybridHit]:
        merged = reciprocal_rank_fusion(
            [("dense", dense_hits), ("lexical", lexical_hits)], k=self.rrf_k
        )
        out: list[HybridHit] = []
        for entry in merged.values():
            item = entry["item"]
            payload = getattr(item, "payload", {}) or {}
            hit = self._base_hit(payload, entry["id"])
            hit.fused_score = entry["fused_score"]
            hit.retrieved_by = entry["retrieved_by"]
            out.append(hit)
        _attach_raw_scores(out, dense_hits, lexical_hits)
        out.sort(key=lambda h: h.fused_score, reverse=True)
        return out[: max(1, top_k)]

    def _weighted_fuse(
        self, dense_hits: list[SearchResult], lexical_hits: list[LexicalHit], top_k: int
    ) -> list[HybridHit]:
        d_scores = _minmax([h.score for h in dense_hits])
        l_scores = _minmax([h.score for h in lexical_hits])
        acc: dict[str, HybridHit] = {}
        for h, s in zip(dense_hits, d_scores, strict=False):
            payload = h.payload or {}
            cid = payload.get("chunk_id", h.id)
            hit = acc.setdefault(cid, self._base_hit(payload, h.id))
            hit.fused_score += self.dense_weight * s
            hit.dense_score = h.score
            if "dense" not in hit.retrieved_by:
                hit.retrieved_by.append("dense")
        for h, s in zip(lexical_hits, l_scores, strict=False):
            payload = h.payload or {}
            cid = payload.get("chunk_id", h.id)
            hit = acc.setdefault(cid, self._base_hit(payload, h.id))
            hit.fused_score += self.lexical_weight * s
            hit.bm25_score = h.score
            if "lexical" not in hit.retrieved_by:
                hit.retrieved_by.append("lexical")
        out = sorted(acc.values(), key=lambda h: h.fused_score, reverse=True)
        return out[: max(1, top_k)]


def _attach_raw_scores(
    hits: list[HybridHit], dense_hits: list[SearchResult], lexical_hits: list[LexicalHit]
) -> None:
    """Annotate fused hits with their raw dense/lexical scores when present."""
    dmap = {(h.payload or {}).get("chunk_id", h.id): h.score for h in dense_hits}
    lmap = {h.id: h.score for h in lexical_hits}
    for hit in hits:
        hit.dense_score = float(dmap.get(hit.chunk_id, 0.0))
        hit.bm25_score = float(lmap.get(hit.chunk_id, 0.0))
