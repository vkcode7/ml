

# Re-Ranking

Rerankers refine the results you get from a vector database query. They do this by calculating relevance scores for each document based on how important it is to satisfy the user's query. That score is used to reorder the queried documents and return only the top-n results.

Usually, rerankers accept a larger than average number of documents and return the highly relevant subset of those.

Adding a reranker to your vector search or RAG pipelines is an easy way to increase the quality of retrieved documents.

Using a reranker in your search pipeline has a few benefits:

- Increases Recall: When using the same number of returned documents, reranked results often contain more relevant documents than when using semantic search alone.
- Increases Precision: As relevance scoring happens during query time, it contextualizes the query and initial set of retrieved documents. This new ordering ensures the most relevant documents are prioritized.
- Improves User Experience: Having a tractable number of highly relevant results reduces the time to benefit from the search for the user, impacting churn/conversion metrics.

RAG pipelines benefit from a reranking step during document retrieval, which ensures the context window has the smallest number of highly relevant documents for use. 

Pinecone Rerank uses the bge-reranker-v2-m3 model, a lightweight open-source and multilingual reranker.

https://www.pinecone.io/learn/refine-with-rerank/


```python
import os
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import pymupdf  # PyMuPDF
from typing import List, Dict

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 1. PDF Loading & Chunking ──────────────────────────────────────────────────

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Extract text from PDF and split into overlapping chunks."""
    doc = pymupdf.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 20:  # skip tiny fragments
                continue
            chunks.append({
                "id": len(chunks),
                "text": " ".join(chunk_words),
                "page": page_num + 1,
                "source": Path(pdf_path).name,
            })
    
    doc.close()
    print(f"✅ Loaded {len(chunks)} chunks from {len(doc)} pages")
    return chunks

# ── 2. Embedding & Vector Store (in-memory) ───────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Embed texts using OpenAI in batches (handles large PDFs)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
        print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
    return np.array(all_embeddings, dtype=np.float32)

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm

def retrieve_top_k(query: str, chunks: List[Dict], embeddings: np.ndarray, top_k: int = 20) -> List[Dict]:
    """Retrieve top-k chunks via dense retrieval."""
    q_emb = np.array(client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding, dtype=np.float32)
    
    scores = cosine_similarity(q_emb, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["retrieval_score"] = float(scores[idx])
        results.append(chunk)
    return results

# ── 3. Re-Ranking ─────────────────────────────────────────────────────────────

def rerank_with_llm(query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Re-rank candidates using an LLM to score relevance.
    For production, use a dedicated cross-encoder (e.g., Cohere Rerank).
    """
    numbered = "\n\n".join(
        f"[{i}] (Page {c['page']})\n{c['text'][:300]}..."
        for i, c in enumerate(candidates)
    )
    
    prompt = f"""You are a relevance judge. Score each passage 0-10 for how well it answers the query.
Return ONLY a JSON array of objects with "index" and "score" keys.

Query: {query}

Passages:
{numbered}

JSON scores:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        # handle {"scores": [...]} or just [...]
        scores_list = data if isinstance(data, list) else next(iter(data.values()))
        score_map = {item["index"]: item["score"] for item in scores_list}
    except Exception as e:
        print(f"⚠️ Re-rank parse error: {e}. Using retrieval scores.")
        score_map = {}
    
    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = score_map.get(i, chunk["retrieval_score"] * 10)
    
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]

# ── 4. Answer Generation ───────────────────────────────────────────────────────

def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in context_chunks
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. Answer based ONLY on the provided context. "
                "Cite page numbers when possible."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ── 5. Full RAG Pipeline ───────────────────────────────────────────────────────

class PDFRagPipeline:
    def __init__(self, pdf_path: str):
        print(f"\n📄 Indexing: {pdf_path}")
        self.chunks = load_and_chunk_pdf(pdf_path)
        print("🔢 Generating embeddings...")
        self.embeddings = embed_texts([c["text"] for c in self.chunks])
        print("✅ Pipeline ready!\n")
    
    def query(self, question: str, retrieve_k: int = 20, rerank_n: int = 5) -> str:
        print(f"🔍 Retrieving top-{retrieve_k} candidates...")
        candidates = retrieve_top_k(question, self.chunks, self.embeddings, top_k=retrieve_k)
        
        print(f"🔄 Re-ranking to top-{rerank_n}...")
        top_chunks = rerank_with_llm(question, candidates, top_n=rerank_n)
        
        print(f"💬 Generating answer from pages: {[c['page'] for c in top_chunks]}")
        answer = generate_answer(question, top_chunks)
        return answer


# ── Example Usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your PDF path
    PDF_PATH = "your_document.pdf"
    
    rag = PDFRagPipeline(PDF_PATH)
    
    questions = [
        "What are the main conclusions of this document?",
        "Summarize the key findings.",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {q}")
        print(f"{'='*60}")
        answer = rag.query(q)
        print(f"📝 Answer:\n{answer}")
```

## How it works

**Pipeline stages:**

1. **Chunking** — PDF pages are extracted via `PyMuPDF` and split into overlapping 500-word chunks (overlap preserves context across boundaries). Works for PDFs with thousands of pages.

2. **Dense Retrieval** — All chunks are embedded with `text-embedding-3-small` in batches (safe for large docs). At query time, the top-20 most similar chunks are retrieved via cosine similarity.

3. **Re-Ranking** — A lightweight LLM (`gpt-4o-mini`) scores the 20 candidates for relevance to the query and returns the top-5. This is the key improvement over naive RAG — it catches retrieval misses and re-orders results semantically.

4. **Generation** — The top-5 re-ranked chunks (with page numbers) are passed as context to generate a grounded answer.

## Install dependencies

```bash
pip install openai pymupdf numpy
```

## For production at scale

| Component | Upgrade to |
|---|---|
| Vector store | Pinecone, Weaviate, or pgvector |
| Re-ranker | Cohere Rerank API (faster & cheaper) |
| Embeddings | Batch API for cost savings |
| Chunking | Semantic chunking (e.g., `langchain`) |


# Here are the best locally-runnable re-ranking models, organized by use case:

## Dedicated Cross-Encoders (Best for Re-ranking)

These are purpose-built cross-encoders from **sentence-transformers** — the gold standard for local re-ranking:

| Model | Size | Notes |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB | Best speed/quality tradeoff, most popular |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130MB | Slightly better quality than L-6 |
| `cross-encoder/ms-marco-electra-base` | ~440MB | Higher quality, slower |
| `BAAI/bge-reranker-base` | ~280MB | Excellent, multilingual-friendly |
| `BAAI/bge-reranker-large` | ~560MB | Best quality in this family |
| `BAAI/bge-reranker-v2-m3` | ~570MB | State-of-the-art, multilingual |

**Usage:**
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_local(query: str, candidates: list, top_n: int = 5) -> list:
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)
    
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
```

Install: `pip install sentence-transformers`

---

## LLM-Based Re-rankers (Higher Quality, Heavier)

If you want LLM-grade re-ranking fully local via **Ollama**:

| Model | Size | Notes |
|---|---|---|
| `mxbai-rerank-base-v1` | ~280MB | Ollama-native re-ranker |
| `mxbai-rerank-large-v1` | ~560MB | Best Ollama re-ranker |
| Any small LLM (Llama 3.2, Phi-3, Gemma 2B) | 2–8GB | Same LLM-scoring trick as the OpenAI example above |

**Ollama usage:**
```python
import ollama

def rerank_ollama(query: str, candidates: list, top_n: int = 5) -> list:
    scored = []
    for chunk in candidates:
        resp = ollama.chat(model="mxbai-rerank-large-v1", messages=[
            {"role": "user", "content": f"Query: {query}\nPassage: {chunk['text']}\nRelevance score 0-10:"}
        ])
        score = float(resp["message"]["content"].strip().split()[0])
        scored.append({**chunk, "rerank_score": score})
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
```

---

## My Recommendations

**For most projects:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Tiny (80MB), fast (CPU-friendly), great quality, zero API cost, drop-in replacement

**For best quality:** `BAAI/bge-reranker-v2-m3`
- State-of-the-art scores on BEIR benchmarks, multilingual, still runs on CPU

**For GPU setups:** `BAAI/bge-reranker-large` or any Ollama LLM
- Much faster inference if you have even a modest GPU

The `sentence-transformers` cross-encoders are the sweet spot — they're specifically trained for the query/passage relevance task (unlike repurposing a chat LLM), are extremely fast, and completely free to run locally.
