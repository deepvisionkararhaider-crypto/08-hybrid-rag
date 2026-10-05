"""Shared text utilities: tokenization and a small English stopword list.

The lexical components (hashing embedder, rule-based query expander, extractive
generator) score token overlap. Filtering stopwords removes high-frequency noise
words that would otherwise dominate similarity.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers him his
    how i if in into is it its itself just me more most my no nor not of off on
    once only or other our ours out over own same she should so some such than that
    the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with
    you your yours
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (no stopword removal)."""
    return _TOKEN_RE.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed -- the lexical signal used for scoring."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
