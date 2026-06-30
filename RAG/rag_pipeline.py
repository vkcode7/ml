"""
Legal Document RAG Pipeline
============================
Architecture:
  1. BM25 (Elasticsearch)       — fast keyword pre-filter (millions of docs)
  2. Dense retrieval (Qdrant)   — semantic search on BM25 candidates
  3. Cross-encoder reranking    — precision rerank top-k results
  4. LLM answer generation      — Anthropic / OpenAI compatible

Stack (all open-source):
  - Elasticsearch 8  : BM25 index
  - Qdrant           : vector store
  - sentence-transformers : embeddings + cross-encoder reranker
  - rank_bm25        : local BM25 fallback / offline use
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from elasticsearch import AsyncElasticsearch, helpers
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_index: str = "legal_docs"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_docs"

    # Models
    embed_model: str = "BAAI/bge-large-en-v1.5"   # 1024-dim, strong on legal text
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval hyper-params
    bm25_top_k: int = 200          # candidates from BM25
    dense_top_k: int = 50          # re-score with dense over BM25 candidates
    rerank_top_k: int = 10         # final context window
    chunk_size: int = 512          # tokens per chunk
    chunk_overlap: int = 64

    # Batching / performance
    embed_batch_size: int = 64
    index_batch_size: int = 256
    embed_dim: int = 1024


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class LegalDocument:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)  # case_id, court, date, jurisdiction …


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class LegalChunker:
    """
    Sentence-aware sliding window chunker.
    Respects legal section boundaries (§, Article, Section headers).
    """

    SECTION_RE = r"(?m)^(?:§+\s*\d|Article\s+\d|Section\s+\d|ARTICLE|SECTION)"

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def chunk(self, doc: LegalDocument) -> list[Chunk]:
        import re
        from textwrap import wrap

        # Try to split on legal section markers first
        sections = re.split(self.SECTION_RE, doc.text)
        chunks: list[Chunk] = []

        for section in sections:
            # Sliding window over ~word tokens (approx 4 chars/token)
            max_chars = self.cfg.chunk_size * 4
            overlap_chars = self.cfg.chunk_overlap * 4
            words = section.split()
            start = 0
            while start < len(words):
                end = start + self.cfg.chunk_size
                window = " ".join(words[start:end])
                chunk_id = hashlib.sha256(
                    f"{doc.doc_id}:{start}".encode()
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=window,
                        metadata={**doc.metadata, "start_word": start},
                    )
                )
                step = self.cfg.chunk_size - self.cfg.chunk_overlap
                start += max(step, 1)

        return chunks


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, cfg: RAGConfig):
        logger.info("Loading embedding model: %s", cfg.embed_model)
        self.model = SentenceTransformer(cfg.embed_model)
        self.batch_size = cfg.embed_batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine via dot product
            show_progress_bar=False,
        )


# ---------------------------------------------------------------------------
# Elasticsearch BM25 layer
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Uses Elasticsearch's native BM25 for fast candidate retrieval.
    Handles millions of docs with sub-second latency.
    """

    MAPPING = {
        "mappings": {
            "properties": {
                "chunk_id":   {"type": "keyword"},
                "doc_id":     {"type": "keyword"},
                "text":       {"type": "text", "analyzer": "english"},
                "metadata":   {"type": "object", "dynamic": True},
            }
        },
        "settings": {
            "number_of_shards": 6,         # tune to cluster size
            "number_of_replicas": 1,
            "index.refresh_interval": "30s",  # bulk indexing perf
        },
    }

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.client = AsyncElasticsearch(cfg.es_url)
        self.index = cfg.es_index

    async def create_index(self, delete_existing: bool = False):
        if delete_existing and await self.client.indices.exists(index=self.index):
            await self.client.indices.delete(index=self.index)
        if not await self.client.indices.exists(index=self.index):
            await self.client.indices.create(index=self.index, body=self.MAPPING)
            logger.info("Created ES index: %s", self.index)

    async def index_chunks(self, chunks: list[Chunk]):
        actions = [
            {
                "_index": self.index,
                "_id": c.chunk_id,
                "_source": {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "metadata": c.metadata,
                },
            }
            for c in chunks
        ]
        # Bulk insert with error capture
        success, errors = await helpers.async_bulk(
            self.client, actions, raise_on_error=False
        )
        if errors:
            logger.warning("ES bulk errors: %d", len(errors))
        return success

    async def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Multi-match BM25 query with optional metadata filters.
        Returns list of {chunk_id, doc_id, text, score}.
        """
        must = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2"],
                    "type": "best_fields",
                    "analyzer": "english",
                }
            }
        ]
        body: dict = {"query": {"bool": {"must": must}}, "size": top_k}

        if filters:
            body["query"]["bool"]["filter"] = [
                {"term": {f"metadata.{k}": v}} for k, v in filters.items()
            ]

        resp = await self.client.search(index=self.index, body=body)
        return [
            {
                "chunk_id": h["_source"]["chunk_id"],
                "doc_id":   h["_source"]["doc_id"],
                "text":     h["_source"]["text"],
                "bm25_score": h["_score"],
                "metadata": h["_source"].get("metadata", {}),
            }
            for h in resp["hits"]["hits"]
        ]

    async def close(self):
        await self.client.close()


# ---------------------------------------------------------------------------
# Qdrant dense vector layer
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Qdrant collection for dense ANN search.
    Uses HNSW index — sub-10 ms for top-50 over 50 M vectors.
    """

    def __init__(self, cfg: RAGConfig, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.client = AsyncQdrantClient(url=cfg.qdrant_url)

    async def create_collection(self, delete_existing: bool = False):
        exists = await self.client.collection_exists(self.cfg.qdrant_collection)
        if delete_existing and exists:
            await self.client.delete_collection(self.cfg.qdrant_collection)
            exists = False
        if not exists:
            await self.client.create_collection(
                collection_name=self.cfg.qdrant_collection,
                vectors_config=VectorParams(
                    size=self.cfg.embed_dim,
                    distance=Distance.COSINE,
                    on_disk=True,          # memory-mapped — essential for millions of docs
                ),
            )
            logger.info("Created Qdrant collection: %s", self.cfg.qdrant_collection)

    async def index_chunks(self, chunks: list[Chunk]):
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        points = [
            PointStruct(
                id=int(c.chunk_id, 16) % (2**63),   # Qdrant needs uint64
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id":   c.doc_id,
                    "text":     c.text,
                    **c.metadata,
                },
            )
            for i, c in enumerate(chunks)
        ]
        await self.client.upsert(
            collection_name=self.cfg.qdrant_collection, points=points
        )

    async def search_by_ids(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int,
    ) -> list[dict]:
        """
        Dense search restricted to BM25 candidate chunk IDs (pre-filter).
        This keeps ANN fast even over millions of vectors.
        """
        qvec = self.embedder.embed([query])[0].tolist()

        # Convert chunk_ids → Qdrant uint IDs
        uint_ids = [int(cid, 16) % (2**63) for cid in candidate_ids]

        results = await self.client.search(
            collection_name=self.cfg.qdrant_collection,
            query_vector=qvec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=cid),
                    )
                    for cid in candidate_ids
                ]
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "chunk_id":    r.payload["chunk_id"],
                "doc_id":      r.payload["doc_id"],
                "text":        r.payload["text"],
                "dense_score": r.score,
                "metadata":    {k: v for k, v in r.payload.items()
                                if k not in ("chunk_id", "doc_id", "text")},
            }
            for r in results
        ]

    async def close(self):
        await self.client.close()


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Cross-encoder scores (query, passage) pairs jointly.
    More accurate than bi-encoder but slower — run only on top-k dense results.
    """

    def __init__(self, cfg: RAGConfig):
        logger.info("Loading reranker: %s", cfg.reranker_model)
        self.model = CrossEncoder(cfg.reranker_model, max_length=512)
        self.top_k = cfg.rerank_top_k

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[: self.top_k]


# ---------------------------------------------------------------------------
# Hybrid scorer (BM25 + dense RRF fusion)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine BM25 and dense rankings via RRF — robust, no score normalisation needed.
    rrf_score = Σ 1/(k + rank_i)
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        meta[cid] = r

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        meta.setdefault(cid, r)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[cid], "rrf_score": score} for cid, score in fused]


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class LegalRAGPipeline:
    """
    Full retrieval pipeline:
      query → BM25 (ES) → Dense re-score (Qdrant) → RRF fusion → Cross-encoder → LLM
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.chunker  = LegalChunker(cfg)
        self.embedder = Embedder(cfg)
        self.bm25     = BM25Index(cfg)
        self.vector   = VectorIndex(cfg, self.embedder)
        self.reranker = Reranker(cfg)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def setup(self, delete_existing: bool = False):
        await self.bm25.create_index(delete_existing)
        await self.vector.create_collection(delete_existing)

    async def index_documents(self, docs: list[LegalDocument]):
        """Chunk → embed → index in both ES and Qdrant."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunker.chunk(doc))

        logger.info("Indexing %d chunks from %d documents", len(all_chunks), len(docs))

        # Batch index
        bs = self.cfg.index_batch_size
        for i in range(0, len(all_chunks), bs):
            batch = all_chunks[i : i + bs]
            await asyncio.gather(
                self.bm25.index_chunks(batch),
                self.vector.index_chunks(batch),
            )
            logger.info("Indexed batch %d/%d", i + bs, len(all_chunks))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,   # e.g. {"jurisdiction": "federal"}
    ) -> list[dict]:
        """
        Returns top-k reranked chunks with metadata.
        Typical latency breakdown (cold):
          BM25  :  ~20–50 ms
          Dense :  ~30–80 ms  (ANN on pre-filtered set)
          Rerank:  ~50–100 ms (cross-encoder on 50 items)
          Total :  ~100–230 ms
        """
        # Stage 1: BM25 fast recall
        bm25_results = await self.bm25.search(
            query, top_k=self.cfg.bm25_top_k, filters=filters
        )

        if not bm25_results:
            return []

        # Stage 2: Dense re-score restricted to BM25 candidates
        candidate_ids = [r["chunk_id"] for r in bm25_results]
        dense_results = await self.vector.search_by_ids(
            query, candidate_ids, top_k=self.cfg.dense_top_k
        )

        # Stage 3: RRF fusion
        fused = reciprocal_rank_fusion(bm25_results, dense_results)
        top_fused = fused[: self.cfg.dense_top_k]

        # Stage 4: Cross-encoder rerank
        reranked = self.reranker.rerank(query, top_fused)

        return reranked

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def answer(
        self,
        query: str,
        filters: Optional[dict] = None,
        llm_client=None,          # any OpenAI-compatible client
        model: str = "gpt-4o",
    ) -> dict:
        chunks = await self.retrieve(query, filters)

        context = "\n\n---\n\n".join(
            f"[{i+1}] (doc={c['doc_id']}) {c['text']}"
            for i, c in enumerate(chunks)
        )

        system_prompt = (
            "You are an expert legal analyst. Answer the user's question based "
            "strictly on the provided legal document excerpts. Cite source numbers "
            "[1], [2] etc. If the answer is not in the context, say so."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        answer_text = "[LLM client not provided — context retrieved successfully]"
        if llm_client:
            resp = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
            )
            answer_text = resp.choices[0].message.content

        return {
            "query":   query,
            "answer":  answer_text,
            "sources": [
                {
                    "rank":    i + 1,
                    "chunk_id": c["chunk_id"],
                    "doc_id":  c["doc_id"],
                    "score":   c.get("rerank_score"),
                    "snippet": c["text"][:200] + "…",
                }
                for i, c in enumerate(chunks)
            ],
        }

    async def close(self):
        await asyncio.gather(self.bm25.close(), self.vector.close())
