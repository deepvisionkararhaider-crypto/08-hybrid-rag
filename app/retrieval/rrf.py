"""Reciprocal Rank Fusion (RRF) — shared across the portfolio.

RRF combines several ranked lists into one ranking using only **rank position**:

    score(d) = Σ_lists  1 / (k + rank_of_d_in_list)

Because it ignores raw scores, it merges lists whose scores are on incomparable
scales (BM25 vs cosine vs cross-encoder logits) without normalisation. Used by
Hybrid RAG (08), Reranker RAG (09), Enterprise RAG (20) and Production RAG (45).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Ranked(Protocol):
    """Anything with an id and a score, as produced by a vector or lexical store."""

    id: str
    score: float


def reciprocal_rank_fusion(
    result_lists: Sequence[tuple[str, Sequence[Any]]],
    k: int = 60,
    *,
    key: str = "chunk_id",
) -> dict[str, dict[str, Any]]:
    """Fuse ranked lists into ``{id: {fused_score, best_score, retrieved_by, item}}``.

    ``result_lists`` is a sequence of ``(label, results)`` where each result has an
    ``id`` and a ``score`` and a ``payload`` dict (or is a plain dict containing
    ``key``). Ranks are 1-based.
    """
    merged: dict[str, dict[str, Any]] = {}

    def _id_of(item: Any) -> str:
        payload = getattr(item, "payload", None)
        if isinstance(payload, dict):
            return payload.get(key) or getattr(item, "id", "")
        if isinstance(item, dict):
            return item.get(key) or item.get("id", "")
        return getattr(item, "id", "")

    def _score_of(item: Any) -> float:
        return float(getattr(item, "score", item.get("score", 0.0) if isinstance(item, dict) else 0.0))

    for label, results in result_lists:
        for rank, item in enumerate(results, start=1):
            ident = _id_of(item)
            if not ident:
                continue
            entry = merged.get(ident)
            if entry is None:
                entry = {
                    "id": ident,
                    "fused_score": 0.0,
                    "best_score": _score_of(item),
                    "retrieved_by": [label],
                    "item": item,
                }
                merged[ident] = entry
            else:
                entry["best_score"] = max(entry["best_score"], _score_of(item))
                if label not in entry["retrieved_by"]:
                    entry["retrieved_by"].append(label)
            entry["fused_score"] += 1.0 / (k + rank)

    return merged


@dataclass
class FusionEntry:
    id: str
    fused_score: float
    best_score: float
    retrieved_by: list[str] = field(default_factory=list)
    item: Any = None


def ranked_entries(
    result_lists: Sequence[tuple[str, Sequence[Any]]], k: int = 60, *, key: str = "chunk_id"
) -> list[FusionEntry]:
    """RRF convenience wrapper returning a sorted list of :class:`FusionEntry`."""
    merged = reciprocal_rank_fusion(result_lists, k=k, key=key)
    return sorted(
        (
            FusionEntry(
                id=e["id"], fused_score=e["fused_score"], best_score=e["best_score"],
                retrieved_by=e["retrieved_by"], item=e["item"],
            )
            for e in merged.values()
        ),
        key=lambda x: x.fused_score,
        reverse=True,
    )
