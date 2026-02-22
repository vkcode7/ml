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

# SFT

**Supervised Fine-Tuning (SFT)** is the process of taking a pretrained model and continuing to train it on a labeled dataset of `(input, desired_output)` pairs — teaching the model to behave in a specific way.

Here's a complete, runnable example using **Hugging Face Transformers + a small GPT-2 model** so it works on a laptop without a GPU:The most important conceptual pieces to understand:

**The -100 label trick** is the heart of SFT. You concatenate `prompt + completion` into one sequence, but mask the prompt tokens with `-100` so PyTorch's loss function ignores them. The model only learns to predict the *completion* — not to memorize the prompt. Without this, the model would waste capacity learning to reproduce the question, and would also learn to "hallucinate" continuations of arbitrary text.

```
Tokens : [What] [is] [capital] ... [Answer:] [Tokyo] [.]
Labels : [-100] [-100] [-100]  ... [-100]    [Tokyo] [.]  ← loss here only
```

**Why SFT works** — the pretrained model already knows facts about the world from pretraining. SFT doesn't teach new knowledge, it teaches *format and behavior*: "when asked a question in this style, respond in this style." That's why InstructGPT only needed ~13K human-labeled examples to dramatically change GPT-3's behavior.

**The pipeline in production** typically goes:
1. **Pretrain** on trillions of tokens (internet text) → raw language model
2. **SFT** on thousands of (prompt, good response) pairs → instruction follower
3. **RLHF / DPO** using human preference rankings → further alignment

This script covers step 2 in its purest form.

```py
"""
Supervised Fine-Tuning (SFT) Demo
==================================
Fine-tunes GPT-2 (small, ~117M params) on a tiny Q&A dataset to teach it
a specific response style — the same concept used to turn a raw pretrained
LLM into an instruction-following assistant (think GPT-3 → InstructGPT).

Concepts demonstrated:
  1. Formatting data as prompt+completion pairs (the "SFT format")
  2. Tokenising with causal-LM labels (only the completion is trained on)
  3. A standard PyTorch training loop
  4. Comparing base vs fine-tuned model outputs

Install:
    pip install torch transformers datasets

Run:
    python sft_demo.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# 1.  TRAINING DATA
#     Format: {"prompt": ..., "completion": ...}
#     In real SFT you'd have thousands of these, carefully human-labelled.
# ---------------------------------------------------------------------------
SFT_DATA = [
    {"prompt": "What is the capital of France?",
     "completion": "The capital of France is Paris."},
    {"prompt": "What is the capital of Germany?",
     "completion": "The capital of Germany is Berlin."},
    {"prompt": "What is the capital of Japan?",
     "completion": "The capital of Japan is Tokyo."},
    {"prompt": "What is the capital of Australia?",
     "completion": "The capital of Australia is Canberra."},
    {"prompt": "What is the capital of Brazil?",
     "completion": "The capital of Brazil is Brasília."},
    {"prompt": "What is the capital of Canada?",
     "completion": "The capital of Canada is Ottawa."},
    {"prompt": "What is the capital of Italy?",
     "completion": "The capital of Italy is Rome."},
    {"prompt": "What is the capital of Spain?",
     "completion": "The capital of Spain is Madrid."},
]

# ---------------------------------------------------------------------------
# 2.  DATASET
#     Key insight: we only want the model to LEARN the completion tokens,
#     not the prompt tokens. We achieve this by setting prompt token labels
#     to -100, which PyTorch's CrossEntropyLoss ignores.
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.samples = []
        for item in data:
            # Separator between prompt and completion
            prompt_text     = item["prompt"] + "\n### Answer:\n"
            completion_text = item["completion"] + tokenizer.eos_token

            prompt_ids     = tokenizer.encode(prompt_text)
            completion_ids = tokenizer.encode(completion_text)

            input_ids = prompt_ids + completion_ids
            # -100 masks prompt tokens — loss only flows through completion
            labels    = [-100] * len(prompt_ids) + completion_ids

            # Truncate / pad to max_length
            input_ids = input_ids[:max_length]
            labels    = labels[:max_length]

            padding = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding
            labels    += [-100] * padding                   # ignore padding too

            self.samples.append({
                "input_ids":      torch.tensor(input_ids),
                "labels":         torch.tensor(labels),
                "attention_mask": torch.tensor(
                    [1] * (max_length - padding) + [0] * padding
                ),
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ---------------------------------------------------------------------------
# 3.  GENERATE HELPER  (used to compare before/after fine-tuning)
# ---------------------------------------------------------------------------
def generate(model, tokenizer, prompt: str, max_new_tokens=40) -> str:
    full_prompt = prompt + "\n### Answer:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 4.  MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # --- Load base model & tokenizer --------------------------------------
    MODEL_NAME = "gpt2"          # swap for "gpt2-medium" / "gpt2-large" etc.
    print(f"Loading {MODEL_NAME}…")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    # --- Baseline output (before fine-tuning) -----------------------------
    TEST_PROMPT = "What is the capital of Japan?"
    print("=" * 60)
    print("BEFORE fine-tuning")
    print(f"  Prompt : {TEST_PROMPT}")
    print(f"  Output : {generate(model, tokenizer, TEST_PROMPT)}")
    print("=" * 60)

    # --- Prepare dataset & dataloader -------------------------------------
    dataset    = SFTDataset(SFT_DATA, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- Training loop ----------------------------------------------------
    EPOCHS    = 10      # small dataset → more epochs needed
    LR        = 5e-5    # standard SFT learning rate (much lower than pretraining)

    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()

    print(f"\nFine-tuning for {EPOCHS} epochs…")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,          # HF computes cross-entropy internally
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  loss: {avg_loss:.4f}")

    # --- Post-fine-tuning output ------------------------------------------
    model.eval()
    print("\n" + "=" * 60)
    print("AFTER fine-tuning")
    for item in SFT_DATA[:4]:   # spot-check a few examples
        prompt = item["prompt"]
        output = generate(model, tokenizer, prompt)
        print(f"  Prompt   : {prompt}")
        print(f"  Expected : {item['completion']}")
        print(f"  Got      : {output}")
        print()
    print("=" * 60)

    # --- Save the fine-tuned model ----------------------------------------
    SAVE_PATH = "./sft_gpt2_capitals"
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}/")
    print("Reload later with:")
    print(f"  model     = GPT2LMHeadModel.from_pretrained('{SAVE_PATH}')")
    print(f"  tokenizer = GPT2Tokenizer.from_pretrained('{SAVE_PATH}')")


if __name__ == "__main__":
    main()
```

Great question — the answer is **yes, but not by training the weights yourself**. There are two distinct paths:

---

**OpenAI's Fine-Tuning API (the practical path)**

OpenAI offers SFT on several of their models through their API. You upload a JSONL file of training examples, kick off a job, and they handle all the distributed training infrastructure on their end. You never touch the weights directly.

As of early 2026, the models available for fine-tuning include **gpt-4o** and **gpt-4o-mini**, among others. You'd want to verify the current list at `platform.openai.com/docs/guides/fine-tuning` since it changes frequently.

The format is simple:
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "What is the capital of Japan?"},
               {"role": "assistant", "content": "The capital of Japan is Tokyo."}]}
{"messages": [{"role": "user", "content": "Summarise this in one sentence."},
               {"role": "assistant", "content": "..."}]}
```

And the API call to launch training:
```python
from openai import OpenAI
client = OpenAI()

# 1. Upload training file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 2. Start fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-2024-08-06"   # check docs for latest fine-tuneable checkpoint
)

print(job.id)   # poll this to track progress
```

---

**Why you can't fine-tune GPT-4 yourself locally**

GPT-4 is estimated to be in the range of hundreds of billions of parameters. To fine-tune a model that size you'd need:

- Dozens to hundreds of A100/H100 GPUs just to *hold the weights* in memory
- Techniques like DeepSpeed ZeRO, FSDP, or model parallelism to distribute across machines
- Weeks of engineering work before a single training step runs

Even with **LoRA** (Low-Rank Adaptation, the most popular efficient fine-tuning technique), which freezes most weights and only trains small adapter matrices, you'd still need massive GPU clusters for a model that size. LoRA is what makes fine-tuning *70B open-source models* feasible on a few GPUs — but GPT-4 is likely several times larger than that.

---

**The practical decision tree**

| Situation | Recommendation |
|---|---|
| Need GPT-4 quality, custom behavior | OpenAI Fine-Tuning API |
| Cost-sensitive, GPT-4o-mini quality is enough | Fine-tune gpt-4o-mini via API (much cheaper) |
| Want full control over weights | Fine-tune an open-source model (Llama 3, Mistral, Qwen) with LoRA on your own GPU |
| Tiny budget, just prompt engineering | Few-shot prompting or RAG instead of SFT |

---

**Honest take**

For most real use cases, fine-tuning via the API is the right call if you need GPT-4 class performance with custom behavior. But many teams find that a well-prompted GPT-4o with RAG outperforms a fine-tuned smaller model anyway — so it's worth benchmarking before committing to SFT at all. SFT shines most when you need consistent *format/style/tone* rather than new *knowledge*.

Yes, exactly — on OpenAI's servers, and you have no direct access to them. Here's the full picture:

**What actually happens when you fine-tune via the API**

OpenAI takes their base model weights, runs your training data through their infrastructure, and the resulting updated weights are stored on their servers under a unique model ID that looks something like:

```
ft:gpt-4o-2024-08-06:your-org-name:your-suffix:abc123xyz
```

You use that string exactly like a normal model name in API calls:

```python
response = client.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:your-org:my-model:abc123xyz",
    messages=[{"role": "user", "content": "..."}]
)
```

That's it — you never see a weight file, a checkpoint, or any binary. It's a fully opaque hosted model.

---

**What you own vs what OpenAI owns**

| Thing | Who controls it |
|---|---|
| Your training data (the JSONL) | You |
| The fine-tuned weight delta | OpenAI holds it, you can delete it |
| The base model weights | OpenAI, always |
| The model ID to call it | You (scoped to your org) |

OpenAI's terms say your fine-tuned model isn't shared with other customers, and you can delete it via the API anytime. But you cannot export the weights, port them to another provider, or run them locally. Ever.

---

**How it likely works under the hood**

OpenAI almost certainly doesn't store a full copy of GPT-4o's weights for every customer's fine-tuned model — that would be prohibitively expensive at scale. The dominant technique for this is **LoRA (Low-Rank Adaptation)**:

Instead of updating all hundreds of billions of parameters, LoRA freezes the base model and only trains small adapter matrices that are a tiny fraction of the total size. Conceptually:

```
Final output = base_model_weights (frozen) + your_lora_adapter (small, trained)
```

So OpenAI likely stores one copy of the giant base model, and a small per-customer adapter on top. At inference time they merge them. This is speculation — OpenAI hasn't publicly confirmed their exact implementation — but it's the industry-standard approach and makes economic sense at their scale.

---

**The deeper implication**

This is a meaningful **vendor lock-in risk** that many teams don't think about until it's too late. If OpenAI deprecates your fine-tuned model checkpoint (which they do periodically — they give notice but it does happen), raises prices, or you want to switch to a competitor, your fine-tuned model is gone. You keep your training data, but you'd have to re-run fine-tuning from scratch on the new provider.

This is one of the strongest arguments for fine-tuning open-source models (Llama, Mistral, Qwen) on your own infrastructure instead — you own the weights completely, can run them anywhere, and are never at the mercy of a third party's deprecation schedule.
