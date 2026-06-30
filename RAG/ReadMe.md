## Scalable RAG system for legal documents with BM25

## Architecture

```
Query
  │
  ▼
Elasticsearch BM25  (top-200 from millions of docs — ~20–50 ms)
  │
  ▼
Qdrant ANN dense search  (re-score BM25 candidates only — ~30–80 ms)
  │
  ▼
RRF Fusion  (Reciprocal Rank Fusion — combines BM25 + dense rankings)
  │
  ▼
Cross-encoder Reranker  (precise joint scoring of top-50 — ~50–100 ms)
  │
  ▼
LLM  (~100–230 ms total retrieval latency before LLM)
```

---

## Key design decisions

**Elasticsearch for BM25** — handles hundreds of millions of docs natively, sharded, with metadata filters (`jurisdiction`, `court`, `year`). Exact keyword matching is critical for legal citations like "17 U.S.C. § 512".

**Qdrant pre-filtered ANN** — dense search runs only over BM25 candidates (not all vectors), keeping ANN latency flat regardless of corpus size. `on_disk=True` enables billion-scale with manageable RAM.

**RRF fusion** — normalisation-free combination of BM25 and dense rankings. More robust than weighted sum since scores aren't on the same scale.

**Cross-encoder reranking** — runs `(query, passage)` pairs through a model that sees both simultaneously. ~10–20% precision lift over bi-encoder alone, run only on the top-50 so latency stays low.

**Legal-aware chunker** — splits on `§`, `Article`, `Section` markers before falling back to sliding window, so legal structure isn't broken mid-clause.

**`LocalRAG` class** (`examples.py`) — uses `rank_bm25` in-process for development with no external services needed.

---

## Quick start

```bash
# 1. Start services
docker-compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Run local example (no Docker needed)
python examples.py

# 4. Production ingestion
# Edit bulk_ingest_from_disk() in examples.py with your data path
```

**Self-hosted Elasticsearch is free.** The basic "Free and Open" tier is available at no cost, which covers everything needed for the RAG pipeline (BM25 indexing, sharding, filtering).

**It's also open source again (as of 2024).** In September 2024, Elastic added the AGPLv3 license as an option alongside existing licenses, making Elasticsearch officially open source again under an OSI-approved license. The client libraries remain under Apache 2.0.

**The main caveat — AGPL.** If you use it under AGPLv3 and expose it as a networked service, your own application code may need to be open-sourced too. For internal tooling this is a non-issue; for commercial SaaS it's worth reviewing with your legal team. Alternatively, you can use the Elastic License 2.0 (ELv2), which is more permissive for most internal/commercial uses but restricts reselling Elasticsearch itself as a managed service.

**Paid tiers exist** for enterprise features (SSO, advanced security, ML features), but none of those are needed for our pipeline.

**If you want a fully permissive alternative**, consider **OpenSearch** (Apache 2.0, no caveats) — it's a fork maintained by AWS/Linux Foundation and is a drop-in replacement for our use case. The pipeline code would require minimal changes.

For BM25 / full-text search at scale, here are the strongest alternatives:

**Drop-in / easiest swap**

- **OpenSearch** (Apache 2.0) — the AWS-maintained fork of Elasticsearch from 2021. Virtually identical API, so the pipeline code needs almost no changes. Fully permissive license, no AGPL concern. Best choice if you want to just swap ES out.

**Purpose-built for hybrid search (BM25 + dense)**

- **Weaviate** — vector DB with native BM25 built in, so it can replace *both* Elasticsearch and Qdrant in the pipeline. One service instead of two. Apache 2.0 for self-hosted.
- **Milvus** — high-performance vector DB with sparse (BM25) + dense hybrid search support. Designed for billions of vectors. Apache 2.0.
- **Qdrant** (already in the pipeline) — actually added sparse vector support recently, so it can handle BM25-style retrieval natively too, potentially eliminating the need for a separate ES layer.

**Lightweight / embedded (good for dev or smaller corpora)**

- **Typesense** — simple, fast, open source (GPL-3). Easy to self-host. BM25-based with vector search added. Not battle-tested at 100M+ doc scale.
- **Tantivy** — Rust-based Lucene equivalent, embedded (no server). Powers Qdrant's sparse search internally. Great if you want zero infra overhead.
- **SQLite FTS5 + sqlite-vec** — surprisingly capable for moderate scale (<10M docs). Zero infra, single file.

**Recommendation for your legal RAG use case:**

| Goal | Best pick |
|---|---|
| Minimal code change from current pipeline | **OpenSearch** |
| Collapse ES + Qdrant into one service | **Weaviate** or **Milvus** |
| Absolute max scale (billions of docs) | **Milvus** |
| Dev / testing, zero infra | **rank_bm25** (already in the code) |

OpenSearch is the most pragmatic swap — same query DSL, same Python client, Apache 2.0 license, and proven at large scale. You'd only need to change the URL in the config and swap `elasticsearch[async]` for `opensearch-py` in requirements.
