"""Tests for the BM25 lexical retriever."""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index


def _index():
    docs = [
        ("c1", "ERR-4032 authentication token expired gateway", {"chunk_id": "c1", "text": "a"}),
        ("c2", "downstream timeout dependency health dashboard", {"chunk_id": "c2", "text": "b"}),
        ("c3", "right sizing instances commitment discounts orphaned volumes", {"chunk_id": "c3", "text": "c"}),
    ]
    idx = BM25Index()
    idx.add([d[0] for d in docs], [d[1] for d in docs], [d[2] for d in docs])
    return idx


def test_exact_term_outranks_others():
    idx = _index()
    hits = idx.search("ERR-4032", k=3)
    assert hits
    assert hits[0].id == "c1"


def test_ranking_is_descending():
    idx = _index()
    hits = idx.search("timeout dashboard", k=3)
    assert hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing():
    idx = _index()
    assert idx.search("", k=3) == []
    assert idx.search("   ", k=3) == []


def test_no_match_returns_empty():
    idx = _index()
    assert idx.search("zebra quasar nebula", k=3) == []


def test_clear_resets_index():
    idx = _index()
    idx.clear()
    assert idx.count() == 0
    assert idx.search("ERR-4032", k=3) == []
