"""API tests for Hybrid RAG (offline, deterministic)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline import get_pipeline
from app.main import app


@pytest.fixture()
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed(client, docs):
    for d in docs:
        assert client.post("/ingest", json={"text": d["text"], "source_id": d["source_id"]}).status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["fusion"] == "rrf"


def test_ingest(client, sample_docs):
    r = client.post("/ingest", json={"text": sample_docs[0]["text"], "source_id": "err.md"})
    assert r.status_code == 200
    assert r.json()["num_chunks"] >= 1


def test_ingest_empty_rejected(client):
    assert client.post("/ingest", json={"text": "", "source_id": "x"}).status_code == 422


def test_ingest_file(client):
    r = client.post("/ingest/file", files={"file": ("a.md", b"# T\n\nERR-9999 failure.", "text/markdown")})
    assert r.status_code == 200


def test_documents(client, sample_docs):
    _seed(client, sample_docs)
    ids = {d["source_id"] for d in client.get("/documents").json()}
    assert "error_codes.md" in ids


def test_query_reports_both_retrievers(client, sample_docs):
    _seed(client, sample_docs)
    r = client.post("/query", json={"question": "ERR-4032 expired token"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["fusion"] == "rrf"
    assert body["dense_candidates"] >= 1
    assert body["lexical_candidates"] >= 1
    assert body["unique_after_fusion"] == len(body["sources"])
    for s in body["sources"]:
        assert s["retrieved_by"]
        assert "fused_score" in s


def test_query_empty_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_without_docs_graceful(client):
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_root(client):
    assert client.get("/").json()["service"] == "hybrid-rag"
