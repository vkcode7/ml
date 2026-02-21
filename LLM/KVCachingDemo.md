**KV caching** is a fundamental optimization in how transformer models (like GPT-4) generate text. To understand it, you need to know a tiny bit about how attention works.

---

**The attention mechanism in one sentence**

At every layer of a transformer, each token looks at every other token before it and computes how much to "attend" to it. To do that, each token produces three vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**.

- The **Query** is essentially "what am I looking for?"
- The **Keys** are "what does each previous token offer?"
- The **Values** are the actual content that gets blended together based on the Q×K similarity scores

---

**The expensive part — without caching**

Say your prompt is 10,000 tokens long. When the model generates token 10,001, it needs the K and V vectors for all 10,000 previous tokens across all its layers. If you then generate token 10,002, it needs those same 10,000 K/V vectors *again*, plus one new one. Without caching, you'd recompute all of them from scratch every single step — enormously wasteful.

---

**What KV caching does**

It simply saves ("caches") those K and V matrices after computing them the first time, and reuses them for every subsequent generation step. The Q vectors still change (the new token has a new query), but the K and V vectors for all *previous* tokens are frozen and reused.

```
Prompt tokens [1 … 10,000]  →  compute K,V once, store in memory
Generate token 10,001        →  reuse stored K,V + compute new Q
Generate token 10,002        →  reuse stored K,V + compute new Q
...
```

This is why generation feels fast after the initial "thinking" pause — the slow part is processing the prompt (filling the KV cache), and the fast part is the actual token-by-token generation that follows.

---

**What OpenAI's "Prompt Caching" feature adds on top**

Standard KV caching happens *within a single request* — it's always been there. What OpenAI (and Anthropic) added in late 2024 is **cross-request KV caching**: they keep the KV cache alive on their servers between separate API calls. So if your next request starts with the same 10,000-token prefix, they skip recomputing those K/V matrices entirely — the work from your previous request is reused. That's the 50% cost discount and latency reduction you see in practice.

---

**Intuitive analogy**

Think of the model reading a book before answering your question. Without cross-request caching, it re-reads the entire book from page 1 for every new question. With it, the book is already "in working memory" — you just ask the new question and it answers immediately.

Here's a self-contained Python example demonstrating OpenAI's Prompt Caching feature:The script is ready. Here's what it demonstrates and why each part matters:


**How OpenAI Prompt Caching works**

OpenAI automatically caches the KV state of any prompt prefix that is ≥ 1,024 tokens. There's no special API flag — you just repeat the same prefix and the cache is used transparently. The response's `usage.prompt_tokens_details.cached_tokens` field tells you how many tokens were served from cache.

**What the script does**

1. Builds a long, stable system prompt (~1,500+ tokens) containing a "knowledge base" — simulating the real-world pattern of a large document, policy file, or tool schema.
2. Sends **Call 1** — cache miss, populates the cache.
3. Sends **Call 2** with a *different user question* but the *identical system prompt prefix* — expects a cache hit.
4. Prints latency, total prompt tokens, and `cached_tokens` for both calls, plus the dollar savings using gpt-4o's 50% cached-token discount.

**Key design principle for maximising cache hits**

```
[stable system prompt / context]  ← cache this
[variable user message]           ← changes every turn
```

Keep everything that stays the same at the *beginning* of the message list (system prompt, documents, few-shot examples), and put the dynamic part at the end. OpenAI's cache key is a token prefix, so even a single changed token early in the prompt invalidates the whole cache.

**To run:**
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
python prompt_caching_demo.py
```

```py
"""
OpenAI Prompt Caching Demo
==========================
Demonstrates how OpenAI automatically caches the KV state of long prompts,
reducing latency and cost on repeated calls that share a common prefix.

How it works:
- Caching kicks in automatically for prompts >= 1024 tokens (as of late 2024).
- The cache key is the exact token prefix. As long as the first N tokens of
  your prompt are identical across requests, OpenAI reuses the cached KV state.
- No special API flag is needed — it's transparent.
- The response usage object reports `prompt_tokens_details.cached_tokens` so
  you can verify the cache was hit.

Pricing impact (gpt-4o as of early 2026):
  - Normal input tokens:  $2.50 / 1M tokens
  - Cached input tokens:  $1.25 / 1M tokens  (50 % discount)
  - Output tokens:        $10.00 / 1M tokens  (unchanged)

Run:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    python prompt_caching_demo.py
"""

import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Build a long, stable system prompt (must be >= 1024 tokens to be cacheable).
# In production this would be your documentation, policy text, large tool
# schema, few-shot examples, etc.
# ---------------------------------------------------------------------------
LONG_SYSTEM_PROMPT = """
You are an expert software architect and educator specialising in distributed
systems, cloud-native applications, and Python best practices.

=== KNOWLEDGE BASE ===

## Section 1: CAP Theorem
The CAP theorem (Brewer, 2000) states that a distributed data store can
guarantee at most two of the following three properties simultaneously:
  • Consistency   – every read receives the most recent write or an error.
  • Availability  – every request receives a (non-error) response.
  • Partition tolerance – the system continues despite network partitions.

In practice, partition tolerance is non-negotiable for real networks, so
designers must choose between CP (e.g. HBase, Zookeeper) and AP (e.g.
Cassandra, DynamoDB in eventual-consistency mode) systems.

## Section 2: SOLID Principles
  S – Single Responsibility Principle  : a class/module has one reason to change.
  O – Open/Closed Principle            : open for extension, closed for modification.
  L – Liskov Substitution Principle    : subtypes must be substitutable for their base types.
  I – Interface Segregation Principle  : prefer many specific interfaces to one general one.
  D – Dependency Inversion Principle   : depend on abstractions, not concretions.

## Section 3: Twelve-Factor App
  1.  Codebase        – one codebase, many deploys.
  2.  Dependencies    – explicitly declare and isolate.
  3.  Config          – store config in environment.
  4.  Backing services– treat as attached resources.
  5.  Build/Release/Run – strictly separate stages.
  6.  Processes       – execute as stateless processes.
  7.  Port binding    – export services via port binding.
  8.  Concurrency     – scale out via the process model.
  9.  Disposability   – fast startup, graceful shutdown.
  10. Dev/prod parity – keep dev, staging, and prod similar.
  11. Logs            – treat as event streams.
  12. Admin processes – run management tasks as one-off processes.

## Section 4: Python Async IO
asyncio is Python's built-in framework for writing single-threaded concurrent
code using coroutines. Key concepts:
  • Event loop   – orchestrates coroutine execution.
  • Coroutine    – declared with `async def`; suspended with `await`.
  • Task         – wraps a coroutine and schedules it on the loop.
  • Gather       – runs multiple coroutines concurrently.
  • Semaphore    – limits concurrent access to a resource.

## Section 5: Common Design Patterns
  Creational : Singleton, Factory, Abstract Factory, Builder, Prototype.
  Structural : Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy.
  Behavioural: Chain of Responsibility, Command, Iterator, Mediator, Memento,
               Observer, State, Strategy, Template Method, Visitor.

## Section 6: Database Indexing
  • B-Tree index   – default in PostgreSQL/MySQL; good for range queries.
  • Hash index     – O(1) equality lookups; no range support.
  • GIN/GiST       – full-text search, JSONB, geometric data (PostgreSQL).
  • Covering index – includes all columns needed by a query (index-only scan).
  • Partial index  – indexes only rows that satisfy a WHERE clause.
  • Composite      – multi-column; column order matters for prefix matching.

## Section 7: HTTP/2 vs HTTP/3
  HTTP/2 : multiplexing over a single TCP connection; header compression (HPACK);
           server push; still subject to head-of-line blocking at TCP level.
  HTTP/3 : built on QUIC (UDP-based); eliminates TCP head-of-line blocking;
           faster handshake (0-RTT resumption); independent stream delivery.

## Section 8: Security Best Practices
  • Store passwords with bcrypt/argon2 (never plain SHA-* alone).
  • Rotate secrets; use a vault (HashiCorp Vault, AWS Secrets Manager).
  • Validate and sanitise all input; parameterise SQL queries.
  • Apply least-privilege IAM roles.
  • Enable CSP, HSTS, and secure cookie flags for web apps.
  • Pin dependencies and scan for CVEs (Dependabot, Snyk, pip-audit).

=== END KNOWLEDGE BASE ===

Answer questions using only the knowledge above. Be concise and precise.
""" * 2   # repeat twice to comfortably exceed 1024 tokens


def call_model(user_question: str, call_label: str) -> dict:
    """Send a completion request and return timing + usage info."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LONG_SYSTEM_PROMPT},
            {"role": "user",   "content": user_question},
        ],
        max_tokens=120,
        temperature=0,
    )
    elapsed = time.perf_counter() - start

    usage = response.usage
    # `prompt_tokens_details` is the key attribute that reveals cache hits
    details = usage.prompt_tokens_details  # PromptTokensDetails object

    print(f"\n{'='*60}")
    print(f"  {call_label}")
    print(f"{'='*60}")
    print(f"  Latency              : {elapsed:.2f}s")
    print(f"  Prompt tokens        : {usage.prompt_tokens}")
    print(f"    ↳ cached tokens    : {details.cached_tokens}")   # <-- the magic number
    print(f"    ↳ non-cached tokens: {usage.prompt_tokens - details.cached_tokens}")
    print(f"  Completion tokens    : {usage.completion_tokens}")
    print(f"  Answer               : {response.choices[0].message.content.strip()[:200]}")

    return {
        "label": call_label,
        "latency_s": elapsed,
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": details.cached_tokens,
    }


def main():
    print(__doc__)
    print("\nEstimated system-prompt tokens: large (>= 1024 — caching eligible)")
    print("\nSending two requests that share the same long system-prompt prefix.")
    print("The first request POPULATES the cache; the second HITS it.\n")

    q1 = "In one sentence, what does the 'L' in SOLID stand for?"
    q2 = "In one sentence, what is HTTP/3 built on top of?"

    r1 = call_model(q1, "Call 1  – cache MISS expected (populates cache)")
    # Brief pause — the cache is available almost immediately, but a small
    # delay makes the demo easier to reason about.
    time.sleep(1)
    r2 = call_model(q2, "Call 2  – cache HIT expected (same system-prompt prefix)")

    # ----------- Summary ---------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Call 1 cached tokens : {r1['cached_tokens']}  (expect ~0)")
    print(f"  Call 2 cached tokens : {r2['cached_tokens']}  (expect large positive number)")

    saved = r2['cached_tokens']
    if saved > 0:
        # gpt-4o pricing as of early 2026
        normal_cost   = saved / 1_000_000 * 2.50
        cached_cost   = saved / 1_000_000 * 1.25
        print(f"\n  Cache hit saved {saved} tokens on the second call.")
        print(f"  Without cache : ${normal_cost:.6f}")
        print(f"  With cache    : ${cached_cost:.6f}")
        print(f"  Saved         : ${normal_cost - cached_cost:.6f}  (50% discount)")
    else:
        print("\n  No cached tokens reported — the cache may not have been warm yet.")
        print("  Try running the script again; caches are typically ready within seconds.")

    print("\nKey takeaway:")
    print("  Keep your system prompt (or large context) as a stable prefix.")
    print("  Vary only the *user* message across calls to maximise cache reuse.")


if __name__ == "__main__":
    main()
```

# RAG and KV Caching

RAG is actually one of the *best* use cases for prompt caching. Here's how it maps:

**Typical RAG pattern without caching awareness:**
```
[system prompt]  +  [retrieved chunks]  +  [user question]
```
If the retrieved chunks change with every query (because each question retrieves different documents), your cache hit rate will be low.

**Optimised RAG pattern for caching:**
```
[system prompt + full knowledge base]  +  [user question]
```
Instead of dynamically retrieving different chunks per query, you stuff the *entire* knowledge base (or a large stable subset) into the system prompt once. Every call shares the same massive prefix → near-100% cache hit rate after the first call.

This is sometimes called **"retrieval-free RAG"** or **"long-context RAG"**, and it's become increasingly practical in 2025 because:

- **Context windows are huge now** — gpt-4o supports 128K tokens, Gemini 1.5 Pro supports 1M tokens, so fitting an entire domain's worth of docs is often feasible.
- **Cached tokens are cheap** — at 50% discount, stuffing 100K tokens of docs into every call costs the same as 50K uncached tokens.
- **Latency stays low** — the KV cache means the model doesn't reprocess all that context from scratch each time.

**When traditional (dynamic) RAG still wins:**

- Your knowledge base is genuinely massive (millions of documents) and can't fit in context.
- Different users need completely different subsets of data — no stable shared prefix exists.
- You need very precise, citation-level retrieval rather than "the model scans everything."

**Hybrid approach** (common in production):

```
[system prompt]                     ← always cached
[broad topic context / top-N docs]  ← semi-stable, cached when possible  
[user question]                     ← varies
```

You retrieve a *larger, coarser* set of chunks (e.g. always pull all docs for a given customer or product category) rather than hyper-precise per-query retrieval. This keeps the prefix stable enough to cache while still scoping the context meaningfully.

The mental model is: **the cache boundary is at the first token that changes**. So you want everything stable pushed as far left (toward the top) as possible, and everything dynamic pushed as far right (toward the end) as possible.
