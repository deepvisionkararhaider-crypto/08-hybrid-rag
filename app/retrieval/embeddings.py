"""Embedding backends behind a single :class:`Embedder` interface.

Two implementations (same contract as Project 01 so the portfolio shares one
embedding abstraction):

* :class:`SentenceTransformerEmbedder` -- real semantic embeddings from a Hugging
  Face ``sentence-transformers`` model (e.g. ``all-MiniLM-L6-v2``). This is the
  production path and requires ``pip install -r requirements-ml.txt``.
* :class:`HashingEmbedder` -- a deterministic, dependency-free, fully offline
  feature-hashing embedder. It is *lexical*, not semantic, but stable and instant,
  which makes it ideal for tests and for running the whole pipeline with zero ML
  downloads.

``build_embedder("auto")`` prefers the real model and transparently falls back to
hashing when ``sentence-transformers`` is unavailable, so the app always runs.

All embedders return L2-normalized float32 matrices, which lets the vector store
use a plain inner product as cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import numpy as np

from app.core.logging import get_logger
from app.core.text_utils import content_tokens

logger = get_logger(__name__)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    name: str = "base"
    model_name: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` L2-normalized float32 matrix."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Signed feature-hashing (hashing-trick) embedder with sublinear tf scaling.

    Deterministic across processes and platforms (uses blake2b, not ``hash()``).
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = "hashing"
        self.model_name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(content_tokens(text))
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "big")
                idx = h % self.dim
                sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
                out[row, idx] += sign * (1.0 + math.log(count))
        return l2_normalize(out)


class SentenceTransformerEmbedder(Embedder):
    """Real semantic embeddings via ``sentence-transformers``."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Loading sentence-transformers model '%s'...", model_name)
        self._model = SentenceTransformer(model_name)
        self.name = "sentence-transformers"
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("Model loaded. Embedding dim=%d", self.dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


def build_embedder(provider: str = "auto", model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Factory selecting an embedder from config.

    ``auto`` tries the real transformer model and falls back to hashing on any
    import/download failure, keeping the service runnable without ML deps.
    """
    provider = (provider or "auto").lower()
    if provider == "hashing":
        return HashingEmbedder()
    if provider in ("sentence-transformers", "st", "transformers"):
        return SentenceTransformerEmbedder(model_name)
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # pragma: no cover - depends on optional deps/network
        logger.warning(
            "sentence-transformers unavailable (%s). Falling back to HashingEmbedder.", exc
        )
        return HashingEmbedder()
