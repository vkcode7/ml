### 🔎 Inference I: KV-Cache (Key–Value Cache)

When a GPT-style transformer generates text **token by token**, it repeatedly uses self-attention.
Without optimization, every new token would recompute attention over *all previous tokens* — very expensive.

**KV-cache solves this.**

---

## 🧠 Quick Intuition

During inference:

1. Each token produces:

   * **Q** (Query)
   * **K** (Key)
   * **V** (Value)

2. For the *next* token:

   * We only need a new **Q**
   * Previous **K and V don’t change**

So instead of recomputing K and V for all prior tokens, we:

> ✅ **Cache K and V once**
> ➜ Reuse them for every next token

That’s the **KV-cache**.

---

## 📦 What Exactly Is Cached?

For each transformer layer:

```
Layer L:
  K_cache: [t1, t2, t3, ..., tn]
  V_cache: [t1, t2, t3, ..., tn]
```

Stored per layer, per head.

So total memory:

```
O(layers × heads × sequence_length × head_dim)
```

This is why long context = large memory usage.

---

## ⚡ Why It Matters

Without KV-cache:

```
Cost per token ≈ O(n²)
```

With KV-cache:

```
Cost per new token ≈ O(n)
```

Where `n` = sequence length so far.

This is what makes fast autoregressive decoding possible.

---

## 🏗 Step-by-Step During Inference

Assume prompt:

> "Hello world"

### Step 1 — Process prompt

* Compute Q, K, V for all tokens
* Store all K and V in cache

### Step 2 — Generate next token

* Compute Q for new token only
* Use:

  ```
  attention(Q_new, K_cached, V_cached)
  ```
* Append new K and V to cache

Repeat.

---

## 🔥 Why KV-Cache Is Inference-Only

During **training**:

* We process full sequences in parallel
* No need for cache

During **inference**:

* We generate sequentially
* Cache avoids recomputation

---

## 🧮 Memory Tradeoff

KV-cache trades:

| Saves   | Costs      |
| ------- | ---------- |
| Compute | GPU memory |

Large context windows (e.g., 128k tokens) are expensive mainly because of KV-cache storage.

---

## 🎯 Simple Mental Model

Think of it like:

> 📒 Instead of re-reading the whole book every time you write the next sentence,
> you keep notes (K,V) from previous pages and only add new notes.

---

If you'd like, I can next explain:

* KV-cache memory calculation example (with numbers)
* Paged KV-cache (used in vLLM)
* FlashAttention interaction
* Multi-query / grouped-query attention optimization
