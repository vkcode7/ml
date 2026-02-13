Here’s a clean, structured map of **fine-tuning techniques** used for LLMs.

We’ll group them by **how much of the model you update**.

---

# 🧱 1️⃣ Full Fine-Tuning (Dense FT)

Update **all parameters** of the model.

```
W ← W - η ∇L
```

### ✅ Pros

* Maximum flexibility
* Best domain adaptation

### ❌ Cons

* Very expensive (VRAM heavy)
* Risk of catastrophic forgetting

Used when:

* You own large compute
* Need deep domain shift

---

# 🧩 2️⃣ Parameter-Efficient Fine-Tuning (PEFT)

Update only a **small subset of parameters**.

Much cheaper.

---

## 🔹 LoRA (Low-Rank Adaptation)

Add low-rank matrices:

```
W' = W + A·B
```

Only train A and B.

### Why it works:

Weight updates are often low-rank.

Most popular PEFT method.

---

## 🔹 QLoRA

LoRA + 4-bit quantization.

* Base model in 4-bit
* LoRA layers in higher precision
* Extremely memory efficient

Allows 65B models on single GPU.

---

## 🔹 Adapters

Insert small MLP layers between transformer blocks.

Only adapters are trained.

More parameters than LoRA, but simple.

---

## 🔹 Prefix Tuning

Instead of changing weights:

Learn virtual tokens prepended to input.

Model stays frozen.

Useful for lightweight customization.

---

## 🔹 Prompt Tuning

Learn continuous embeddings that act like prompts.

Even smaller than prefix tuning.

Very lightweight.

---

# 🎯 3️⃣ Supervised Fine-Tuning (SFT)

Train on:

```
Instruction → Ideal Answer
```

Used for:

* Chat alignment
* Domain adaptation
* Structured output learning

Often combined with LoRA.

---

# 🧠 4️⃣ Reinforcement Learning-Based Methods

After SFT.

---

## 🔹 RLHF (Reinforcement Learning from Human Feedback)

* Train reward model
* Optimize policy via PPO

Complex, expensive.

---

## 🔹 DPO (Direct Preference Optimization)

No reward model.
Optimizes preference pairs directly.

Simpler and now widely used.

---

# 🧮 5️⃣ Continual / Domain Adaptive Pretraining

Before SFT.

Train on domain corpus with next-token objective.

Example:

* Finance documents
* Legal corpus
* Medical text

Improves base knowledge before alignment.

---

# 🔀 6️⃣ Multi-Task Fine-Tuning

Train on multiple tasks simultaneously.

Helps:

* Generalization
* Avoid overfitting

---

# 🧠 7️⃣ Instruction Tuning

Large-scale SFT across many tasks.

Improves zero-shot performance.

---

# 📊 Comparison Table

| Method        | Train Params | Compute   | Flexibility       |
| ------------- | ------------ | --------- | ----------------- |
| Full FT       | 100%         | Very High | Max               |
| LoRA          | ~0.1–1%      | Low       | High              |
| QLoRA         | ~0.1%        | Very Low  | High              |
| Adapters      | Small        | Medium    | High              |
| Prefix        | Tiny         | Very Low  | Medium            |
| Prompt tuning | Tiny         | Very Low  | Lower             |
| DPO           | Small        | Medium    | Alignment-focused |

---

# 🎯 Practical Guidance

If you’re building production systems:

* 🟢 Domain adaptation → LoRA or QLoRA
* 🟢 Chat alignment → SFT + DPO
* 🟢 Low-resource GPU → QLoRA
* 🔵 Large enterprise model → Full FT + RLHF
* 🟡 Lightweight customization → Prefix/Prompt tuning

---

# 🧠 Mental Model

Fine-tuning spectrum:

```
Prompt tuning  →  LoRA  →  Full FT
   (tiny)         (small)    (massive)
```

Tradeoff = flexibility vs cost.

---

If you'd like, I can next explain:

* When RAG is better than fine-tuning
* How to choose LoRA rank
* Hyperparameter strategies
* Catastrophic forgetting
* Fine-tuning for financial LLM systems


Here’s the **real-world usage picture (2025)** — what’s actually widely used in industry and open-source.

---

# 🥇 Most Widely Used Today

## 1️⃣ LoRA (and QLoRA) → ⭐⭐⭐⭐⭐

**Most common fine-tuning method overall.**

Used for:

* Domain adaptation
* Instruction tuning
* Enterprise customization
* Open-source models (LLaMA-family, Mistral, etc.)

Why it dominates:

* Cheap
* Stable
* Easy to implement
* Works very well

QLoRA is especially popular because:

* 4-bit base model
* Train on single GPU
* Excellent cost/performance

👉 If someone says “we fine-tuned a model,”
very often it means **LoRA**.

---

## 2️⃣ Supervised Fine-Tuning (SFT) → ⭐⭐⭐⭐⭐

Nearly universal.

Every production chat model goes through SFT.

Used for:

* Chat formatting
* Structured outputs
* Instruction following
* Domain tone alignment

Even when people say “RLHF model,” it *still started with SFT*.

---

# 🥈 Widely Used but More Specialized

## 3️⃣ DPO (Direct Preference Optimization) → ⭐⭐⭐⭐

Now very common in:

* Open-source alignment
* Mid-size companies
* Preference alignment workflows

Replacing PPO-based RLHF in many pipelines because:

* Simpler
* More stable
* Cheaper

---

## 4️⃣ Domain-Adaptive Pretraining → ⭐⭐⭐

Used in:

* Legal
* Finance
* Medical
* Code models

Large enterprises do this more than startups.

---

# 🥉 Less Common (Production)

## 5️⃣ Full Fine-Tuning → ⭐⭐

Used mostly by:

* Big AI labs
* Large enterprises
* When massive compute available

Expensive and risky (catastrophic forgetting).

---

## 6️⃣ RLHF with PPO → ⭐⭐ (declining)

Still used at frontier labs.

But many organizations moved to DPO or simpler alignment methods.

Complex and compute-heavy.

---

## 7️⃣ Adapters → ⭐⭐

Still used, but LoRA largely replaced them.

---

## 8️⃣ Prefix / Prompt Tuning → ⭐

Used in research.
Rare in serious production LLM systems.

LoRA performs better in practice.

---

# 📊 Realistic Ranking (Industry View)

| Technique            | Industry Usage        |
| -------------------- | --------------------- |
| SFT                  | Extremely common      |
| LoRA                 | Extremely common      |
| QLoRA                | Extremely common      |
| DPO                  | Growing fast          |
| Domain Pretraining   | Moderate              |
| Full FT              | Limited (big players) |
| PPO-based RLHF       | Frontier labs         |
| Prefix/Prompt tuning | Rare                  |

---

# 🎯 What Most Companies Actually Do

Typical modern stack:

```
Base Model
   ↓
Domain-Adaptive Pretraining (optional)
   ↓
SFT (via LoRA/QLoRA)
   ↓
DPO alignment
```

That’s the dominant pattern.

---

# 🧠 Key Insight

The industry shifted toward:

> **Parameter-efficient + simple optimization**

Because:

* GPUs are expensive
* Stability matters
* Iteration speed matters
* Scaling LoRA is easy

---

If you'd like, I can next explain:

* What OpenAI likely uses internally
* What Meta uses for LLaMA
* What startups vs banks typically deploy
* When RAG replaces fine-tuning
* Cost comparison of LoRA vs full FT
