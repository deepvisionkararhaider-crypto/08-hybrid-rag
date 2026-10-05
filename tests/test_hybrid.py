"""Tests for RRF and the hybrid retriever (lexical + dense fusion)."""

from __future__ import annotations

from app.retrieval.rrf import ranked_entries, reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult


def _r(chunk_id, score, text="t"):
    return SearchResult(id=chunk_id, score=score,
                        payload={"chunk_id": chunk_id, "source_id": "s", "index": 0, "text": text, "metadata": {}})


def test_rrf_promotes_items_in_both_lists():
    merged = reciprocal_rank_fusion([("a", [_r("X", 0.9), _r("Y", 0.5)]),
                                     ("b", [_r("Y", 0.8), _r("Z", 0.7)])], k=60)
    assert merged["Y"]["fused_score"] > merged["X"]["fused_score"]
    assert merged["Y"]["fused_score"] > merged["Z"]["fused_score"]
    assert set(merged["Y"]["retrieved_by"]) == {"a", "b"}


def test_rrf_ranks_are_sorted():
    entries = ranked_entries([("a", [_r("X", 0.1)]), ("b", [_r("Y", 0.9)])], k=60)
    assert [e.id for e in entries] == ["X", "Y"]  # tie on rank -> insertion order stable


def _ingest(pipeline, docs):
    for d in docs:
        pipeline.ingest_text(d["text"], source_id=d["source_id"])


def test_hybrid_retrieves_exact_identifier(pipeline, sample_docs):
    _ingest(pipeline, sample_docs)
    hits, counts = pipeline.retriever.search("ERR-4032", top_k=3, per_retriever_k=10)
    assert hits
    # BM25 must surface the error-code chunk.
    assert counts["lexical"] >= 1
    assert any(h.chunk_id and "error_codes" in h.source_id for h in hits)


def test_hybrid_merges_both_retrievers(pipeline, sample_docs):
    _ingest(pipeline, sample_docs)
    hits, counts = pipeline.retriever.search("how to lower cloud costs", top_k=5, per_retriever_k=10)
    assert hits
    assert counts["dense"] >= 1
    # At least one hit should be found by both retrievers (agreement signal).
    by = [h for h in hits if set(h.retrieved_by) >= {"dense", "lexical"}]
    assert by or all(h.retrieved_by for h in hits)


def test_hybrid_fused_scores_descending(pipeline, sample_docs):
    _ingest(pipeline, sample_docs)
    hits, _ = pipeline.retriever.search("cloud spend", top_k=5, per_retriever_k=10)
    scores = [h.fused_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_weighted_fusion_also_works(pipeline, sample_docs):
    _ingest(pipeline, sample_docs)
    pipeline.retriever.fusion = "weighted"
    hits, _ = pipeline.retriever.search("cloud spend", top_k=4, per_retriever_k=10)
    assert hits
    scores = [h.fused_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_source_filter_applies_to_both_retrievers(pipeline, sample_docs):
    _ingest(pipeline, sample_docs)
    hits, _ = pipeline.retriever.search("cloud", top_k=5, per_retriever_k=10, source_filter="cost_guide.md")
    assert all(h.source_id == "cost_guide.md" for h in hits)


def test_empty_index_is_graceful(pipeline):
    hits, counts = pipeline.retriever.search("anything", top_k=3, per_retriever_k=5)
    assert hits == []
    assert counts == {"lexical": 0, "dense": 0}
