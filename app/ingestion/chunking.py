"""Recursive character-based text chunking with overlap (from-scratch).

Reused from the portfolio's Project 01 (vanilla-rag) so the whole portfolio shares
one chunking implementation. Text is split on a hierarchy of separators --
paragraphs, then lines, then sentences, then words, then a hard character cut --
so chunk boundaries respect natural structure whenever possible. Adjacent chunks
share ``chunk_overlap`` characters of context, which materially improves
retrieval recall for answers that straddle a boundary.

Each returned :class:`Chunk` carries its character span in the original document
so citations can point back to an exact location.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _recursive_split(
    text: str, start: int, chunk_size: int, separators: Sequence[str]
) -> list[tuple[str, int, int]]:
    """Split ``text`` into atomic segments no larger than ``chunk_size``.

    Returns tuples of ``(segment_text, abs_start, abs_end)`` where the offsets are
    relative to the original document (``start`` is this substring's offset).
    """
    if len(text) <= chunk_size or not separators:
        return [(text, start, start + len(text))] if text else []

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        out: list[tuple[str, int, int]] = []
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            out.append((piece, start + i, start + i + len(piece)))
        return out

    out = []
    cursor = 0
    for part in text.split(sep):
        part_start = start + cursor
        if part:
            if len(part) <= chunk_size:
                out.append((part, part_start, part_start + len(part)))
            else:
                out.extend(_recursive_split(part, part_start, chunk_size, rest))
        cursor += len(part) + len(sep)
    return out


def _merge_with_overlap(
    atomics: list[tuple[str, int, int]], chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    """Greedily pack atomic segments into chunks, seeding each new chunk with the
    trailing segments of the previous one until ``chunk_overlap`` chars are covered.

    Returns a list of ``(char_start, char_end)`` spans into the original text.
    """
    spans: list[tuple[int, int]] = []
    buf: list[int] = []

    for idx, (_tok, _s, end) in enumerate(atomics):
        if buf:
            prospective_len = end - atomics[buf[0]][1]
            if prospective_len > chunk_size:
                spans.append((atomics[buf[0]][1], atomics[buf[-1]][2]))
                tail_end = atomics[buf[-1]][2]
                seed: list[int] = []
                for prev in reversed(buf):
                    seed.insert(0, prev)
                    if (tail_end - atomics[prev][1]) >= chunk_overlap:
                        break
                buf = seed
        buf.append(idx)

    if buf:
        spans.append((atomics[buf[0]][1], atomics[buf[-1]][2]))
    return spans


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Sequence[str] | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping :class:`Chunk` objects.

    Raises ``ValueError`` on a non-positive size or an overlap >= size (which would
    prevent forward progress). Empty / whitespace-only text yields no chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if not text or not text.strip():
        return []

    seps = list(separators) if separators is not None else DEFAULT_SEPARATORS
    atomics = _recursive_split(text, 0, chunk_size, seps)
    if not atomics:
        return []

    spans = _merge_with_overlap(atomics, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    meta = dict(base_metadata or {})
    for i, (s, e) in enumerate(spans):
        chunk_str = text[s:e].strip()
        if not chunk_str:
            continue
        chunk_meta = dict(meta)
        chunk_meta.update({"char_start": s, "char_end": e})
        chunks.append(Chunk(text=chunk_str, index=i, char_start=s, char_end=e, metadata=chunk_meta))
    return chunks
