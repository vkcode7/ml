"""
Usage examples & local BM25 fallback (no Elasticsearch needed for dev/testing)
"""

import asyncio
import json
from pathlib import Path
from rag_pipeline import (
    LegalDocument,
    LegalRAGPipeline,
    RAGConfig,
)


# ---------------------------------------------------------------------------
# Example 1: Ingest and query (production — ES + Qdrant)
# ---------------------------------------------------------------------------

async def production_example():
    cfg = RAGConfig(
        es_url="http://localhost:9200",
        qdrant_url="http://localhost:6333",
        bm25_top_k=200,
        dense_top_k=50,
        rerank_top_k=10,
    )

    pipeline = LegalRAGPipeline(cfg)
    await pipeline.setup(delete_existing=False)

    # -- Index documents (stream from disk / database in production) --
    docs = [
        LegalDocument(
            doc_id="case_001",
            text="""
            UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK
            
            Section 1. The defendant is hereby ordered to cease and desist all 
            operations in violation of 17 U.S.C. § 512. The Digital Millennium 
            Copyright Act safe harbor provisions do not apply when the service 
            provider has actual knowledge of infringing activity and fails to act 
            expeditiously to remove or disable access to the infringing material.
            
            Section 2. Damages shall be awarded pursuant to 17 U.S.C. § 504(c)(2) 
            for willful infringement, not to exceed $150,000 per work infringed.
            """,
            metadata={"court": "SDNY", "year": 2023, "jurisdiction": "federal",
                      "case_type": "copyright"},
        ),
        LegalDocument(
            doc_id="statute_dmca_512",
            text="""
            17 U.S.C. § 512 - Limitations on liability relating to material online.
            
            (a) Transitory Digital Network Communications. A service provider shall 
            not be liable for monetary relief, or, except as provided in subsection 
            (j), for injunctive or other equitable relief, for infringement of 
            copyright by reason of the provider's transmitting, routing, or providing 
            connections for, material through a system or network controlled or 
            operated by or for the service provider.
            
            (b) System Caching. A service provider shall not be liable for monetary 
            relief for infringement of copyright by reason of the intermediate and 
            temporary storage of material on a system or network controlled or 
            operated by or for the service provider.
            """,
            metadata={"type": "statute", "title": "DMCA", "jurisdiction": "federal"},
        ),
    ]

    await pipeline.index_documents(docs)

    # -- Query with jurisdiction filter --
    result = await pipeline.answer(
        query="What are the DMCA safe harbor requirements for online service providers?",
        filters={"jurisdiction": "federal"},
        llm_client=None,   # replace with openai.AsyncOpenAI() or anthropic client
    )

    print(json.dumps(result, indent=2))
    await pipeline.close()


# ---------------------------------------------------------------------------
# Example 2: Local BM25 fallback (rank_bm25, no external services needed)
# ---------------------------------------------------------------------------

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import Optional


class LocalRAG:
    """
    Lightweight version: rank_bm25 + sentence-transformers + cross-encoder.
    Perfect for development / small corpora (<100k docs on single machine).
    """

    def __init__(
        self,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_top_k: int = 50,
        rerank_top_k: int = 5,
    ):
        self.bm25_top_k   = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.embedder  = SentenceTransformer(embed_model)
        self.reranker  = CrossEncoder(reranker_model, max_length=512)

        self._chunks: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._vectors: Optional[np.ndarray] = None
        self._tokenized: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, docs: list[LegalDocument], chunk_size: int = 400):
        """Index documents in memory."""
        for doc in docs:
            words = doc.text.split()
            for start in range(0, len(words), chunk_size - 64):
                chunk_text = " ".join(words[start: start + chunk_size])
                self._chunks.append({
                    "text":     chunk_text,
                    "doc_id":   doc.doc_id,
                    "metadata": doc.metadata,
                })
                self._tokenized.append(self._tokenize(chunk_text))

        self._bm25 = BM25Okapi(self._tokenized)
        texts = [c["text"] for c in self._chunks]
        self._vectors = self.embedder.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        )
        print(f"Indexed {len(self._chunks)} chunks")

    def query(self, query: str) -> list[dict]:
        assert self._bm25 and self._vectors is not None, "Call .index() first"

        # Stage 1: BM25
        tok_q = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(tok_q)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][: self.bm25_top_k]

        # Stage 2: Dense re-score on BM25 candidates
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        cand_vecs = self._vectors[top_bm25_idx]
        dense_scores = cand_vecs @ q_vec

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(top_bm25_idx):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (60 + rank + 1)
        dense_order = top_bm25_idx[np.argsort(dense_scores)[::-1]]
        for rank, idx in enumerate(dense_order):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (60 + rank + 1)

        fused_idx = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:50]

        # Stage 3: Cross-encoder rerank
        candidates = [self._chunks[i] for i in fused_idx]
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, batch_size=32)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[
            : self.rerank_top_k
        ]


# ---------------------------------------------------------------------------
# Example 3: Bulk ingestion from disk (production)
# ---------------------------------------------------------------------------

async def bulk_ingest_from_disk(
    pipeline: LegalRAGPipeline,
    data_dir: str,
    batch_size: int = 500,
):
    """
    Stream .txt / .json legal files from a directory and ingest in batches.
    Handles millions of documents without loading all into memory.
    """
    import os

    batch: list[LegalDocument] = []
    total = 0

    for fname in Path(data_dir).rglob("*.json"):
        with open(fname) as f:
            raw = json.load(f)

        doc = LegalDocument(
            doc_id=raw.get("id", fname.stem),
            text=raw.get("text", ""),
            metadata=raw.get("metadata", {}),
        )
        batch.append(doc)

        if len(batch) >= batch_size:
            await pipeline.index_documents(batch)
            total += len(batch)
            print(f"Ingested {total} documents…")
            batch.clear()

    if batch:
        await pipeline.index_documents(batch)
        total += len(batch)

    print(f"Done. Total documents indexed: {total}")


# ---------------------------------------------------------------------------
# Run local example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Local (no ES/Qdrant needed)
    rag = LocalRAG(bm25_top_k=50, rerank_top_k=5)

    docs = [
        LegalDocument(
            doc_id="dmca_001",
            text="""
            The Digital Millennium Copyright Act (DMCA) provides safe harbor 
            protections for online service providers under 17 U.S.C. § 512.
            To qualify, providers must: (1) adopt and implement a policy to terminate 
            repeat infringers; (2) accommodate standard technical measures; 
            (3) not receive a financial benefit directly attributable to infringing 
            activity; and (4) expeditiously remove infringing content upon receiving 
            proper notice.
            """,
            metadata={"jurisdiction": "federal", "type": "statute"},
        ),
        LegalDocument(
            doc_id="case_viacom_youtube",
            text="""
            Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2d Cir. 2012).
            The Second Circuit held that the DMCA safe harbor requires actual knowledge 
            of specific and identifiable infringements, not merely general awareness that 
            infringement was occurring on the platform. General knowledge is insufficient 
            to disqualify a service provider from safe harbor protection under § 512(c).
            """,
            metadata={"jurisdiction": "federal", "court": "2nd Circuit", "year": 2012},
        ),
    ]

    rag.index(docs)

    results = rag.query("What knowledge standard applies for DMCA safe harbor?")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] doc={r['doc_id']} score={r['rerank_score']:.3f}")
        print(r["text"][:300])
