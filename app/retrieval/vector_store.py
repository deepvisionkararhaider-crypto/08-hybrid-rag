"""Vector stores behind a single :class:`VectorStore` interface.

* :class:`NumpyStore` -- exact brute-force cosine similarity over a dense matrix.
  Zero extra dependencies, perfect for small/medium corpora and for tests.
* :class:`FaissStore` -- FAISS ``IndexFlatIP`` (inner product on L2-normalized
  vectors == cosine). Used automatically when ``faiss-cpu`` is installed.

Both accept an optional predicate so retrieval can be filtered (e.g. by
``source_id``) without a second pass. Vectors are assumed L2-normalized, so the
inner product is the cosine similarity and higher is better.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

Predicate = Callable[[dict[str, Any]], bool] | None


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    name: str = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[dict]) -> None: ...

    @abstractmethod
    def search(
        self, query: np.ndarray, k: int, predicate: Predicate = None
    ) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


class NumpyStore(VectorStore):
    name = "numpy"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids, vectors, payloads) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        self._ids.extend(ids)
        self._payloads.extend(payloads)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query, k, predicate=None) -> list[SearchResult]:
        if self._matrix is None or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores = (self._matrix @ q.T).ravel()  # cosine (vectors normalized)
        order = np.argsort(-scores)
        results: list[SearchResult] = []
        for i in order:
            payload = self._payloads[i]
            if predicate is not None and not predicate(payload):
                continue
            results.append(SearchResult(id=self._ids[i], score=float(scores[i]), payload=payload))
            if len(results) >= k:
                break
        return results

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._payloads.clear()
        self._matrix = None


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        import faiss  # lazy import; only required when this backend is selected

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []

    def add(self, ids, vectors, payloads) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        self._index.add(vectors)
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query, k, predicate=None) -> list[SearchResult]:
        total = self._index.ntotal
        if total == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        fetch = total if predicate is not None else min(k, total)
        scores, idxs = self._index.search(q, fetch)
        results: list[SearchResult] = []
        for score, i in zip(scores[0], idxs[0], strict=False):
            if i < 0:
                continue
            payload = self._payloads[i]
            if predicate is not None and not predicate(payload):
                continue
            results.append(SearchResult(id=self._ids[i], score=float(score), payload=payload))
            if len(results) >= k:
                break
        return results

    def count(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index.reset()
        self._ids.clear()
        self._payloads.clear()


def build_vector_store(provider: str = "auto", dim: int = 256) -> VectorStore:
    """Factory selecting a vector store from config, preferring FAISS on ``auto``."""
    provider = (provider or "auto").lower()
    if provider == "numpy":
        return NumpyStore(dim)
    if provider == "faiss":
        return FaissStore(dim)
    try:
        return FaissStore(dim)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        logger.warning("faiss unavailable (%s). Falling back to NumpyStore.", exc)
        return NumpyStore(dim)
