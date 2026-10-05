"""Document loaders: turn uploaded bytes into plain text.

Supported: ``.txt`` / ``.md`` / ``.rst`` / ``.csv`` / ``.json`` (any UTF-8 text)
and ``.pdf`` (via ``pypdf``). For PDFs we preserve per-page text so citations can
report a page number. Unknown binary types are decoded leniently so the service
never hard-crashes on ingestion; callers surface a clear error when the extracted
text is empty.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".text", ".rst", ".csv", ".json", ".py", ".yaml", ".yml",
}


@dataclass
class LoadedDocument:
    """Extracted document text plus optional per-page offsets for citations."""

    text: str
    page_spans: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def page_for_offset(self, char_offset: int) -> int | None:
        """Return the 1-based page number containing ``char_offset`` (or None)."""
        for i, (start, end) in enumerate(self.page_spans, start=1):
            if start <= char_offset < end:
                return i
        return None


def _extract_pdf(data: bytes) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - depends on pdf content
            logger.warning("Failed to extract a PDF page: %s", exc)
            page_text = ""
        if not page_text.strip():
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        spans.append((start, cursor))
    return LoadedDocument(
        text="".join(parts), page_spans=spans, metadata={"num_pages": len(spans)}
    )


def load_document(filename: str, data: bytes) -> LoadedDocument:
    """Extract text (and page spans when possible) from raw file bytes."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return LoadedDocument(text=text, metadata={"extension": ext or "txt"})
