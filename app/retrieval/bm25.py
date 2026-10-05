"""BM25 lexical retrieval, implemented from scratch (no external search library).

BM25 scores a document by term overlap with the query, weighting rare terms highly
(IDF) and correcting for document length. It is the classic lexical retriever and
complements dense embeddings: it nails **exact terms** (error codes, product names,
acronyms, identifiers) that a semantic model may blur.

Score:

    BM25(q, d) = Σ_t  IDF(t) · (f(t,d) · (k1 + 1)) /
                        (f(t,d) + k1 · (1 − b + b · |d| / avgdl))

    IDF(t) = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.text_utils import tokenize


@dataclass
class LexicalHit:
    id: str
    score: float
    payload: dict[str, Any]


class BM25Index:
    """In-memory BM25 index over a set of documents (chunks)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._docs: list[Counter[str]] = []  # term frequencies per doc
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()  # document frequency per term

    def add(self, ids: Sequence[str], texts: Sequence[str], payloads: Sequence[dict]) -> None:
        for ident, text, payload in zip(ids, texts, payloads, strict=False):
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._ids.append(ident)
            self._payloads.append(payload)
            self._docs.append(tf)
            self._lengths.append(len(tokens) or 1)
            for term in tf:
                self._df[term] += 1

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._payloads.clear()
        self._docs.clear()
        self._lengths.clear()
        self._df.clear()

    @property
    def avgdl(self) -> float:
        return (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0

    def _idf(self, term: str) -> float:
        n = len(self._ids)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[LexicalHit]:
        if not query.strip() or not self._ids or k <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        avgdl = self.avgdl
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._docs):
            dl = self._lengths[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                s += idf * (f * (self.k1 + 1.0)) / denom
            if s > 0.0:
                scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            LexicalHit(id=self._ids[i], score=float(s), payload=self._payloads[i])
            for s, i in scores[:k]
        ]
