"""Generation providers behind a single :class:`LLMProvider` interface.

Default is :class:`ExtractiveProvider` -- a genuinely offline, key-free generator
that composes a grounded answer from the most question-relevant sentences of the
retrieved passages with inline ``[n]`` citations. This keeps the whole RAG loop
runnable with no external service.

:class:`OpenAIProvider` and :class:`HuggingFaceProvider` perform real LLM
generation and activate only when the matching API key is configured -- swap
``LLM_PROVIDER`` in the environment, no code changes. The same abstraction also
powers LLM-based query expansion (``raw_complete``).

Citation numbering is consistent: ``[n]`` always refers to the n-th context
chunk, which is the n-th entry in the response ``sources``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.text_utils import content_tokens

logger = get_logger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_INLINE_MD_RE = re.compile(r"\*\*|__|~~|[*`>]")
_RULE_RE = re.compile(r"^\s*[-_=]{3,}\s*$")


def _clean_line(line: str) -> str:
    line = _LIST_BULLET_RE.sub("", line)
    line = _INLINE_MD_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line or _MD_HEADING_RE.match(raw) or _RULE_RE.match(raw):
            continue
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        for s in _SENT_SPLIT_RE.split(cleaned):
            s = s.strip()
            if s:
                out.append(s)
    return out


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, contexts: list[str]) -> str:
        """Produce an answer grounded in ``contexts`` (ordered best-first)."""

    def raw_complete(self, prompt: str, max_tokens: int = 256) -> str:
        """Free-form completion used by LLM query expansion.

        The offline provider cannot do this, so it returns an empty string and the
        caller transparently falls back to rule-based expansion.
        """
        raise NotImplementedError(f"{self.name} does not support free-form completion")


class ExtractiveProvider(LLMProvider):
    """Offline extractive summarizer over retrieved passages with citations."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return (
                "I couldn't find any relevant information in the indexed documents "
                "to answer this question."
            )

        q_tokens = set(content_tokens(question))
        candidates: list[tuple[float, int, int, str]] = []
        n_ctx = len(contexts)
        for ci, ctx in enumerate(contexts, start=1):
            rank_bonus = 0.15 * (n_ctx - ci + 1) / n_ctx
            for si, sent in enumerate(_sentences(ctx)):
                s_tokens = set(content_tokens(sent))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens)
                coverage = overlap / (len(q_tokens) or 1)
                complete = 0.05 if sent[-1] in ".!?" else 0.0
                score = coverage + rank_bonus - 0.01 * si + complete
                candidates.append((score, ci, si, sent))

        candidates.sort(key=lambda x: x[0], reverse=True)

        seen: set[str] = set()
        picks: list[tuple[int, int, str]] = []
        for _score, ci, si, sent in candidates:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            picks.append((ci, si, sent))
            if len(picks) >= self.max_sentences:
                break

        if not picks:
            first = _sentences(contexts[0])
            picks = [(1, 0, first[0] if first else contexts[0][:200])]

        picks.sort(key=lambda x: (x[0], x[1]))
        body = " ".join(f"{sent} [{ci}]" for ci, _si, sent in picks)
        return f"Based on the retrieved documents: {body}"

    def raw_complete(self, prompt: str, max_tokens: int = 256) -> str:
        return ""  # triggers rule-based fallback in the query generator


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or None

    def _client(self):
        from openai import OpenAI  # lazy import

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, question: str, contexts: list[str]) -> str:
        ctx_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        ctx_block = ctx_block or "(no context retrieved)"
        system = (
            "You are a precise assistant. Answer using ONLY the numbered context below. "
            "Cite the sources you used inline like [1] or [2]. "
            "If the context is insufficient to answer, say so explicitly."
        )
        user = f"Context:\n{ctx_block}\n\nQuestion: {question}\n\nAnswer:"
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()

    def raw_complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class HuggingFaceProvider(LLMProvider):
    name = "huggingface"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _client(self):
        from huggingface_hub import InferenceClient  # lazy import

        return InferenceClient(model=self.model, token=self.api_key)

    def generate(self, question: str, contexts: list[str]) -> str:
        ctx_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        ctx_block = ctx_block or "(no context retrieved)"
        system = (
            "You are a precise assistant. Answer using ONLY the numbered context below. "
            "Cite sources inline like [1]. If the context is insufficient, say so."
        )
        user = f"Context:\n{ctx_block}\n\nQuestion: {question}\n\nAnswer:"
        resp = self._client().chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    def raw_complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client().chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Select a generation provider from config, validating required secrets."""
    provider = (settings.llm_provider or "extractive").lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )

    if provider in ("huggingface", "hf"):
        if not settings.huggingface_api_key:
            raise ValueError("LLM_PROVIDER=huggingface requires HUGGINGFACE_API_KEY to be set.")
        return HuggingFaceProvider(
            api_key=settings.huggingface_api_key,
            model=settings.huggingface_model,
        )

    return ExtractiveProvider()
