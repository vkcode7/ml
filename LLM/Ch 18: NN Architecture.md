## 🧠 Neural Net Architecture: GPT, LLaMA, and MoE

We’ll break this into 3 parts:

1. **GPT evolution (1 → 4)**
2. **LLaMA architectural improvements (RoPE, RMSNorm, GQA)**
3. **MoE (Mixture of Experts)**

---

# 1️⃣ GPT Architecture (1 → 4)

All GPT models are based on:

> Decoder-only Transformer
> Autoregressive next-token prediction

Core block (repeated N times):

```
Token Embedding
   ↓
[ Self-Attention ]
   ↓
[ MLP / Feedforward ]
   ↓
Residual connections + LayerNorm
```

---

## 🔹 GPT-1 (2018)

* ~117M parameters
* Standard Transformer decoder
* Trained on BookCorpus
* Showed unsupervised pretraining works

Architecture:

* Learned positional embeddings
* LayerNorm
* GELU activation

---

## 🔹 GPT-2 (2019)

* Up to 1.5B parameters
* Larger scale
* Better zero-shot behavior

Still:

* Learned positional embeddings
* Full multi-head attention

Main change: **scale**

---

## 🔹 GPT-3 (2020)

* 175B parameters
* Massive scaling
* Few-shot prompting works

Architecture mostly same as GPT-2
Key change: **model size + data scale**

---

## 🔹 GPT-4

Exact architecture not public.

Likely improvements:

* Better training stability
* Possibly mixture-of-experts
* Multimodal capability
* Improved alignment training

Core principle still:

> Large decoder-only transformer

---

# 🧠 Key Insight About GPT Evolution

Most gains from GPT-1 → GPT-3 were from:

```
Scale > architectural change
```

---

# 2️⃣ LLaMA Architecture Improvements

Meta Platforms introduced LLaMA with architectural refinements for efficiency.

LLaMA is still a decoder-only transformer, but with key improvements.

---

## 🔹 RoPE (Rotary Positional Embeddings)

Problem with learned position embeddings:

* Poor extrapolation beyond training length

RoPE idea:

* Encode position as rotation in embedding space

Instead of:

```
Add position vector
```

RoPE:

```
Rotate Q and K vectors by angle proportional to position
```

Benefits:

* Better long-context generalization
* Relative positional information
* Cleaner mathematical structure

Now standard in most modern LLMs.

---

## 🔹 RMSNorm (Root Mean Square Normalization)

GPT uses LayerNorm.

LLaMA uses RMSNorm.

Difference:

* LayerNorm normalizes mean + variance
* RMSNorm normalizes only variance

Formula intuition:

```
x / sqrt(mean(x²))
```

Benefits:

* Slightly faster
* More stable for very deep networks
* Works well at scale

---

## 🔹 GQA (Grouped Query Attention)

Standard Multi-Head Attention:

Each head has:

```
Q, K, V
```

Memory heavy.

GQA idea:

* Many query heads
* Fewer key/value heads shared

Example:

```
32 Q heads
8 K/V heads
```

Benefits:

* Reduces KV-cache memory
* Faster inference
* Minimal quality loss

Important for long-context models.

---

# 3️⃣ MoE (Mixture of Experts)

Instead of one big MLP per layer:

Use multiple expert MLPs.

```
Input
  ↓
Router (gating network)
  ↓
Select top-k experts
  ↓
Combine outputs
```

---

## 🔥 Why MoE?

Dense model:

```
Every token → every neuron
```

MoE:

```
Every token → small subset of experts
```

So:

* Parameter count huge
* Active compute per token small

Example:

* 8 experts
* Only 2 activated per token

Effective parameters large, compute moderate.

---

## 🧠 Sparse vs Dense

| Type      | Compute         | Parameters |
| --------- | --------------- | ---------- |
| Dense GPT | High            | High       |
| MoE       | Lower per token | Very High  |

---

## ⚠️ Challenges

* Load balancing
* Routing instability
* More complex training
* Distributed systems needed

---

# 🧠 Big Architectural Trends

Modern LLM improvements:

* RoPE (better position encoding)
* RMSNorm (simpler normalization)
* GQA (efficient attention)
* MoE (scaling without linear compute growth)
* FlashAttention (optimized kernel)
* Longer context windows

---

# 🎯 Final Mental Model

GPT = simple dense transformer scaled up
LLaMA = optimized transformer
MoE = sparse scaling strategy

---

If you'd like next, I can explain:

* Attention math from scratch
* Why RoPE works geometrically
* Why MoE sometimes underperforms dense models
* Scaling laws across dense vs MoE
* How architecture affects inference cost
