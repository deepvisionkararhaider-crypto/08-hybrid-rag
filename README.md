# 08 — Hybrid RAG

> **Portfolio project 08 of 50** — *Hybrid RAG*. Combines **lexical BM25** and
> **dense semantic** retrieval and fuses their rankings with **Reciprocal Rank
> Fusion**. **No deployment** — source + local Docker only.

---

## Project

**Hybrid RAG** runs two complementary retrievers over the same chunk set and merges
their results:

- **BM25 (lexical)** — the classic term-frequency retriever, implemented from scratch.
- **Dense (semantic)** — embeddings + cosine similarity (shared with projects 01/06/07).

Their ranked lists are combined with **Reciprocal Rank Fusion (RRF)**, which uses only
rank position and therefore needs no score normalisation across two very different
score scales (BM25's unbounded positive scores vs cosine's `[-1, 1]`).

Every returned source reports its **fused score**, its **raw BM25 and dense scores**,
and **which retriever(s) found it** — so the hybrid contribution is fully visible.

---

## Problem

A single retrieval method has systematic blind spots:

- **Dense-only misses exact terms.** Embeddings compress meaning, so rare identifiers
  — error codes (`ERR-4032`), SKUs, acronyms, version strings, names — get blurred
  into a generic region and can be outranked by topically-similar noise. Ask
  "ERR-4032" and a semantic search may return any chunk about errors.
- **BM25-only misses paraphrase.** Lexical matching fails when the query and document
  share meaning but not words: query "reduce cloud costs" vs document "lower cloud
  spend" share **no** content terms, so BM25 scores that document zero.
- **They fail in opposite directions**, which is exactly why combining them helps.

---

## Solution

Run both, then fuse.

```
                  question
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   BM25 (lexical)           Dense (semantic)
   exact terms, IDF         paraphrase, synonyms
        │                         │
     ranked list              ranked list
        └────────────┬────────────┘
                     ▼
        Reciprocal Rank Fusion (k=60)
        score(d) = Σ 1/(k + rank(d))
                     ▼
        fused ranking → top-k chunks
                     ▼
                   LLM
                     ▼
            Answer + Sources
```

A document found by **both** retrievers accumulates two RRF contributions and is
promoted — agreement between lexical and semantic evidence is a strong relevance
signal. A document found by only **one** retriever still survives (at lower rank),
which is what recovers each retriever's unique wins.

`weighted` fusion (min-max normalise each list, then a weighted sum) is provided as
an alternative and is documented as such.

---

## Architecture

```
08-hybrid-rag/
├── app/
│   ├── api/routes.py                 # /health /ingest /ingest/file /query /documents
│   ├── core/{config,logging,pipeline,text_utils}.py
│   ├── ingestion/{chunking,loaders}.py         # shared
│   ├── retrieval/
│   │   ├── bm25.py                   # ★ BM25 lexical index (from scratch)
│   │   ├── hybrid.py                 # ★ HybridRetriever: BM25 + dense → RRF
│   │   ├── rrf.py                    # shared Reciprocal Rank Fusion
│   │   ├── embeddings.py             # shared (hashing | sentence-transformers)
│   │   ├── vector_store.py           # shared (numpy | faiss)
│   │   └── retriever.py              # dense child search
│   ├── llm/provider.py               # shared LLM abstraction
│   ├── models/schemas.py             # HybridSource exposes bm25_score/dense_score/retrieved_by
│   └── main.py
├── frontend/index.html               # zero-build UI with retriever provenance
├── data/  tests/  scripts/seed.py
├── Dockerfile  docker-compose.yml  pyproject.toml  .env.example
└── README.md  ARCHITECTURE.md
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Workflow

```
Document → chunk → ┬─► embed → dense vector store
                   └─► tokenize → BM25 index
Question
   ↓
BM25 search  +  dense search   (in parallel, same chunk set)
   ↓
Reciprocal Rank Fusion
   ↓
top-k fused chunks
   ↓
LLM
   ↓
Answer + Sources (+ bm25/dense/fused scores + which retriever found each)
```

---

## Technologies

| Concern | Choice | Why |
|---|---|---|
| API | **FastAPI** + pydantic v2 | Portfolio standard |
| Lexical retrieval | **BM25** (from scratch) | Transparent, testable, no external search engine |
| Dense retrieval | **`all-MiniLM-L6-v2`** + **hashing** fallback | Real semantics + deterministic offline tests |
| Vector store | **FAISS** + **numpy** fallback | Exact cosine; runs anywhere |
| Fusion | **RRF (k=60)**, plus `weighted` | Rank-based, scale-free — the standard hybrid merge |
| LLM | **LLMProvider**: `extractive`/`openai`/`huggingface` | Decoupled, env-configured |
| Frontend | Static HTML + vanilla JS | Lightweight, no build step |

**Why RRF as the default fusion:** BM25 scores are unbounded and query-length
dependent; cosine scores are bounded in `[-1, 1]`. Normalising them to combine is
fragile and corpus-dependent. RRF sidesteps the problem by using only rank — it is
the de-facto standard for hybrid retrieval for exactly this reason.

---

## Installation

```bash
git clone https://github.com/deepvisionkararhaider-crypto/08-hybrid-rag.git
cd 08-hybrid-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt      # pytest + ruff
# optional: pip install -r requirements-ml.txt   (real embeddings + FAISS)
```

## Configuration

Copy `.env.example` → `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `BM25_K1` / `BM25_B` | `1.5` / `0.75` | BM25 term-saturation and length-normalisation |
| `PER_RETRIEVER_K` | `10` | Candidates fetched from **each** retriever |
| `FUSION` | `rrf` | `rrf` or `weighted` |
| `RRF_K` | `60` | RRF smoothing constant |
| `DENSE_WEIGHT` / `LEXICAL_WEIGHT` | `0.5` / `0.5` | Used only when `FUSION=weighted` |
| `TOP_K` | `5` | Final fused chunks sent to the LLM |
| `EMBEDDING_PROVIDER` / `VECTOR_STORE` | `auto` | Backend selection |
| `LLM_PROVIDER` | `extractive` | Generation backend |

> **Secrets:** API keys come from the environment/`.env` only and are never committed.

## Running locally

```bash
python scripts/seed.py
uvicorn app.main:app --reload --port 8000
# UI:   http://localhost:8000/ui
# Docs: http://localhost:8000/docs
```

**Docker:**
```bash
docker compose up --build     # → http://localhost:8000 (UI at /ui)
```

## API Documentation

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Config + index size (incl. BM25 params, fusion mode) |
| `POST` | `/ingest` | Ingest text (indexes into **both** retrievers) |
| `POST` | `/ingest/file` | Ingest `.txt` / `.md` / `.pdf` |
| `POST` | `/query` | BM25 + dense → RRF fuse → cited answer |
| `GET`  | `/documents` | List documents |

**Request**
```json
{ "question": "ERR-4032 expired token" }
```

**Response** (abridged)
```json
{
  "question": "ERR-4032 expired token",
  "answer": "Based on the retrieved documents: ... [1]",
  "fusion": "rrf",
  "lexical_candidates": 2,
  "dense_candidates": 10,
  "unique_after_fusion": 5,
  "sources": [
    { "rank": 1, "source_id": "error_codes.md", "fused_score": 0.0325,
      "bm25_score": 3.41, "dense_score": 0.29,
      "retrieved_by": ["dense", "lexical"] }
  ]
}
```

`bm25_score` / `dense_score` / `retrieved_by` make each retriever's contribution
explicit.

## Testing

```bash
pytest -q
pytest tests/test_bm25.py -q      # lexical retriever
pytest tests/test_hybrid.py -q    # fusion + merge
pytest tests/test_api.py -q
```

Deterministic and hermetic — offline backends, no network, no LLM. Tests include a
**lexical-only** case (`ERR-4032` must be found by BM25) and a **paraphrase** case
(dense must contribute), and assert RRF promotes chunks found by both retrievers.

## Limitations

- **RRF discards magnitude.** A chunk that is a *very* strong lexical match but only
  moderately ranked will not dominate — RRF only sees ranks, not scores. `weighted`
  fusion is available when score magnitude matters.
- **BM25 is in-memory** and rebuilt on boot (no persisted inverted index).
- **No reranking** — fusion reorders, it does not re-read (leads into project 09).
- **No query expansion** — hybrid here is single-query (see project 06 for multi-query).
- **Tokenisation is simple** (lowercase alphanumeric), so no stemming/lemmatisation —
  "running" and "run" are distinct terms.
- **Extractive default LLM.**
- **No benchmark claims.** The API reports honest candidate counts; no BEIR-style
  or ablation numbers are claimed because no such experiment was run.

## Future Improvements

- **Cross-encoder rerank** of the fused list (project 09).
- **Combine with multi-query** (project 06) so each generated query runs a hybrid
  search.
- **Tune `dense_weight`/`lexical_weight`** on a labelled set and report real numbers.
- **Stemming + stopword-aware BM25**, and field-weighted BM25 (title vs body).
- **Persist the inverted index**.

---

## Comparison with Portfolio Projects 01–07

| # | Project | Core idea | How 08 differs |
|---|---------|-----------|----------------|
| 01 | Vanilla RAG | dense-only retrieval | 08 adds a lexical retriever alongside |
| 02 | LangChain RAG | pipeline via LangChain | 08 implements BM25 + RRF itself |
| 03 | LlamaIndex RAG | pipeline via LlamaIndex | 08 exposes both retrievers' raw scores |
| 04 | Advanced RAG | retrieval-quality experiments | 08's lever is **two complementary retrievers + fusion** |
| 05 | HyDE RAG | expand the query with a synthetic answer | 08 changes neither query nor embedding — it changes the **retrieval methods** |
| 06 | Multi-Query RAG | many queries → fuse rankings | 06 fuses **runs of one retriever**; 08 fuses **different retriever types** over one query |
| 07 | Parent-Document RAG | search children, answer with parents | 07 changes **granularity**; 08 changes **retrieval modality** |
| **08** | **Hybrid RAG** | **BM25 + dense → RRF** | — |

**Why 08 is distinct:** project 06 fuses multiple *dense* runs; 08 fuses a *lexical*
run with a *dense* run. The failure modes addressed are different: 06 attacks query
phrasing, 08 attacks the fundamental term-vs-meaning blind spot of any single
retriever. Its response exposes `bm25_score`, `dense_score`, `lexical_candidates` and
`dense_candidates` — none of which exist in 01–07.

> No deployment URLs by design: this portfolio task is GitHub/source focused.

---

*Part of a 50-project AI engineering portfolio. Previous: `07-parent-document-rag`.
Next: `09-reranker-rag`.*
