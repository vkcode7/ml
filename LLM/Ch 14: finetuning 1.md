## 🎯 Fine-tuning I: SFT (Supervised Fine-Tuning)

After pretraining (next-token prediction on massive corpora), a base GPT model is powerful — but not aligned for structured tasks or chat.

**SFT = train the model on labeled input → output pairs**
so it learns desired behavior directly.

---

# 🧠 What Is SFT?

You take a pretrained model and train it further on:

```
Instruction / Input  →  Ideal Output
```

Example (chat style):

```
User: Explain bond duration.
Assistant: Bond duration measures sensitivity of price to interest rate changes...
```

The model learns:

* Follow instructions
* Answer in proper format
* Adopt tone (helpful, safe, structured)

---

# 🏗 How It Works Internally

Training objective is still:

```
Predict next token
```

But now on curated data.

If training example is:

```
<user> What is convexity?
<assistant> Convexity measures...
```

Loss is computed only on assistant tokens.

So the model learns:

* Condition on user message
* Produce correct assistant continuation

---

# 💬 Why SFT Enables Chat

Chat formatting is learned via SFT.

Special tokens are used:

```
<system>
<user>
<assistant>
```

The model learns conversational turn-taking patterns.

Without SFT:

* Model continues text
* Not structured chat

With SFT:

* Model responds like an assistant

---

# 📦 Data Format (Typical)

Instruction-tuning dataset:

```json
{
  "messages": [
    {"role": "system", "content": "You are a financial expert."},
    {"role": "user", "content": "Explain yield curve inversion."},
    {"role": "assistant", "content": "..."}
  ]
}
```

This becomes training sequence tokens.

---

# 🔥 Full Fine-Tuning vs PEFT

## 1️⃣ Full Fine-Tuning

Update **all model weights**.

Pros:

* Maximum flexibility
* Best task adaptation

Cons:

* Expensive (VRAM heavy)
* Risk of catastrophic forgetting

---

## 2️⃣ PEFT (Parameter-Efficient Fine-Tuning)

Update **small subset of parameters only**.

Huge memory savings.

Common methods:

* Adapters
* Prefix tuning
* **LoRA**

---

# 🚀 LoRA (Low-Rank Adaptation)

Most popular PEFT method.

Instead of updating large weight matrix:

```
W  (e.g., 4096 × 4096)
```

LoRA learns two small matrices:

```
A (4096 × r)
B (r × 4096)
```

Where `r` is small (like 8 or 16).

Effective weight:

```
W' = W + A·B
```

Only A and B are trained.

---

## 🧠 Why This Works

Weight updates in large models are often low-rank.

So instead of learning full matrix,
we learn a compressed delta.

---

# 📊 Memory Comparison

Example: 7B model

| Method     | Trainable Params |
| ---------- | ---------------- |
| Full FT    | 7B               |
| LoRA (r=8) | ~10–20M          |

Massive reduction.

---

# 🎯 What SFT Actually Teaches

* Instruction following
* Tone and style
* Output formatting
* Domain adaptation
* Structured responses
* Chat behavior

---

# ⚖ Tradeoffs

| Benefit               | Cost                       |
| --------------------- | -------------------------- |
| Aligns model to task  | Needs labeled data         |
| Enables chat          | May overfit small datasets |
| Domain specialization | Can forget general ability |

---

# 🧠 Simple Mental Model

Pretraining = learns language
SFT = learns how to behave

---

If you'd like, I can next explain:

* SFT vs RLHF difference
* How LoRA works mathematically in attention layers
* How to implement LoRA training pipeline
* Practical hyperparameters (rank, alpha, dropout)
* When to use SFT vs RAG (very important design decision)
