# Architecture — Hybrid RAG

## 1. The core idea

Two retrievers with **opposite failure modes** run over the same chunk set, and their
rankings are fused.

```
   INGEST
   ──────
   text ──► chunk ──┬──► embed ──► dense vector store (cosine)
                    └──► tokenize ──► BM25 inverted index      (same chunks, same ids)

   QUERY
   ─────
   question ──┬──► dense search ──► ranked list A
              └──► BM25 search  ──► ranked list B
                                        │
                                        ▼
                     Reciprocal Rank Fusion (k=60)
                     score(d) = Σ_lists 1/(k + rank_list(d))
                                        │
                                        ▼
                                fused ranking → top-k
                                        │
                                        ▼
                                       LLM
                                        │
                                        ▼
                              Answer + Sources
                       (with bm25_score, dense_score, retrieved_by)
```

## 2. Components

### 2.1 `app/retrieval/bm25.py` — lexical retriever (core)

A from-scratch BM25 index. For each document it stores term frequencies and document
length; globally it stores document frequency per term.

```
BM25(q, d) = Σ_t IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
IDF(t)     = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
```

- `k1` controls term-frequency saturation (default 1.5).
- `b` controls length normalisation (default 0.75).
- Scoring is O(#terms × #docs) — fine for this project's scale.

`BM25Index.add(ids, texts, payloads)` keeps payloads so a hit carries its text and
metadata for citations.

### 2.2 `app/retrieval/embeddings.py` + `vector_store.py` — dense backends

Shared, proven implementations: `SentenceTransformerEmbedder` with a deterministic
`HashingEmbedder` fallback; `FaissStore` with a `NumpyStore` fallback. L2-normalised
output means the store's inner product is cosine.

### 2.3 `app/retrieval/retriever.py` — dense search wrapper

`Retriever` indexes and searches chunks densely, with an optional `source_id`
predicate.

### 2.4 `app/retrieval/rrf.py` — fusion (shared)

`reciprocal_rank_fusion(result_lists, k)` merges `(label, results)` lists into
`{id: {fused_score, best_score, retrieved_by, item}}`. It reads only `id`, `score`
and (optionally) `payload` from each item, so it works for both `SearchResult` and
`LexicalHit`. Ranks are 1-based; `k` damps the top ranks' dominance.

### 2.5 `app/retrieval/hybrid.py` — HybridRetriever (core)

- `add(records)` indexes the **same** records into both retrievers (keeps ids stable).
- `search(question, top_k, per_retriever_k, source_filter)`:
  1. runs a dense search and a BM25 search, each fetching `per_retriever_k`;
  2. fuses with `rrf` (default) or `weighted`;
  3. returns `HybridHit`s plus `{"lexical": n, "dense": m}` candidate counts.
- `_rrf_fuse` calls the shared RRF and annotates each hit with raw `bm25_score` /
  `dense_score` via `_attach_raw_scores`.
- `_weighted_fuse` min-max normalises each list and takes
  `dense_weight·dense_norm + lexical_weight·lexical_norm` (documented alternative).

A chunk present in both lists gets two RRF contributions and is promoted; a chunk in
one list survives at lower rank.

### 2.6 `app/core/pipeline.py` — orchestrator

`HybridRAGPipeline` builds embedder → dense store → `Retriever` → BM25 → `HybridRetriever`,
and the LLM. It persists document text + chunk metadata to
`DATA_DIR/documents.json` and rebuilds both indexes on boot (keeping the on-disk
format backend-independent). Per-request timings are recorded.

### 2.7 `app/api/routes.py` + `app/main.py`

`/health`, `/ingest`, `/ingest/file`, `/query`, `/documents`. Pipeline injected via
`Depends(get_pipeline)` for test overriding. Static UI at `/ui`.

## 3. Why RRF is the default fusion

| | BM25 score | Cosine score |
|---|---|---|
| Range | `[0, ∞)` | `[-1, 1]` |
| Depends on | corpus stats, query length, #terms | embedding model, normalisation |
| Comparable across retrievers? | **no** | **no** |

Normalising and summing is possible (`weighted`) but needs tuning per corpus. RRF uses
**rank only**, so it is scale-free and requires no calibration — hence the default.

## 4. Failure modes handled

| Situation | Behaviour |
|---|---|
| Empty index | `/query` returns empty sources; graceful message |
| Empty question | `ValueError` → HTTP 400/422 |
| Query matches no documents lexically | Dense list still returns results |
| Query is an exact identifier | BM25 list carries it; RRF keeps it |
| `faiss`/`sentence-transformers` missing | Fallback to numpy / hashing |
| Corrupt persistence | Logged warning; empty index |
| `source_filter` set | Applied to **both** retrievers before fusion |
