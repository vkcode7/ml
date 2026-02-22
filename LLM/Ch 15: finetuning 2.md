## 🎯 Fine-Tuning II: Reinforcement Learning (RL, RLHF, PPO, DPO)

After **SFT** teaches the model *how to answer*,
**RL fine-tuning** teaches it *which answers humans prefer*.

Think:

> SFT → correctness
> RL → preference alignment

---

# 🧠 Why RL Is Needed

SFT learns from:

```
Prompt → Ideal Answer
```

But many tasks don’t have a single correct answer.

Example:

* More polite vs less polite
* More concise vs verbose
* Safer vs risky
* Financial explanation: retail-friendly vs quant-heavy

Human preference is subjective.

That’s where **Reinforcement Learning from Human Feedback (RLHF)** comes in.

---

# 🔥 RLHF (Reinforcement Learning from Human Feedback)

Pipeline:

### 1️⃣ SFT model

Start from supervised fine-tuned model.

### 2️⃣ Collect preference data

Humans compare two outputs:

```
Prompt: Explain convexity.

Answer A: ...
Answer B: ...
Human: prefers B
```

### 3️⃣ Train a Reward Model

Model learns:

```
R(prompt, answer) → scalar score
```

### 4️⃣ Optimize policy with RL

Update LLM to maximize reward.

---

# 🧮 PPO (Proximal Policy Optimization)

Most famous RLHF algorithm.

Used in early ChatGPT systems.

### Objective:

Maximize:

```
Expected reward
```

But prevent model from drifting too far from SFT.

So we optimize:

```
Reward - KL penalty
```

Where:

* Reward = reward model score
* KL penalty = distance from original SFT model

This keeps model stable.

---

### 🧠 Intuition

Without KL penalty:

* Model may exploit reward model
* Produce weird but high-reward outputs

With KL penalty:

* Stays close to base personality

---

# ⚠️ PPO Challenges

* Complex training
* Two models (policy + reward)
* RL instability
* Expensive
* Sensitive hyperparameters

---

# 🚀 DPO (Direct Preference Optimization)

Modern alternative.

Much simpler.

Instead of:

* Training reward model
* Running PPO

DPO directly optimizes preference pairs.

---

## DPO Idea

Given:

```
Prompt
Chosen answer (preferred)
Rejected answer
```

Optimize model so:

```
P(chosen | prompt) > P(rejected | prompt)
```

With closed-form objective.

No reward model.
No RL loop.
No PPO.

---

# 📊 PPO vs DPO

| Feature        | PPO (RLHF) | DPO    |
| -------------- | ---------- | ------ |
| Reward model   | Yes        | No     |
| RL loop        | Yes        | No     |
| Stability      | Hard       | Easier |
| Compute cost   | High       | Lower  |
| Implementation | Complex    | Simple |

DPO is now widely preferred for open models.

---

# 🏗 Conceptual Math View

### PPO optimizes:

```
max E[R] - β KL(π || π_ref)
```

### DPO optimizes:

```
log σ(β (log π(chosen) - log π(rejected)))
```

Much cleaner.

---

# 🧠 Big Picture of Fine-Tuning Stages

```
Pretraining → language knowledge
SFT → instruction following
RLHF / DPO → alignment to human preferences
```

---

# 🎯 What RL Fine-Tuning Actually Changes

* Politeness
* Helpfulness
* Safety boundaries
* Conciseness vs verbosity
* Tone
* Refusal behavior
* Reasoning style

---

# 🧠 Mental Model

If SFT teaches:

> “How to write answers”

RL teaches:

> “Which answers humans like best”

---

If you'd like next, I can explain:

* Why reward hacking happens
* How KL divergence prevents collapse
* Why DPO often matches PPO in practice
* Alignment vs capability distinction
* How this applies to domain LLMs (e.g., fixed income chat models)
  

# NOTE on OpenAI and RL:

The honest practical answer is: if you're on OpenAI's API, real RL on the model weights isn't available to you today. Your options are best-of-N, prompt optimisation, or waiting to see if OpenAI exposes preference-based fine-tuning (they've hinted at it). If RL over model weights matters to your use case, that's the strongest argument for going open-source with Llama or Qwen, where you own the full pipeline from SFT through DPO/PPO.

# Further READ if time permits:

Yes — and this is actually the exact pipeline used to build modern aligned LLMs. The full stack is called **RLHF (Reinforcement Learning from Human Feedback)** and it sits on top of SFT as a third stage. Let me break down how it applies to both RAG and SFT outcomes.

---

**The full training pipeline in context**

```
Pretraining (on internet text)
       ↓
SFT (on labeled prompt→completion pairs)
       ↓
RL (optimise for human preference / reward signal)
```

RL doesn't replace SFT — it refines it. SFT teaches the model *what kind of responses to give*, RL teaches it *which of those responses are better*.

---

**How RLHF works mechanically**

There are three components:

**1. The SFT model** — your starting point, already fine-tuned.

**2. A Reward Model (RM)** — a separate model trained to score responses. You train it by showing human raters pairs of responses to the same prompt and having them pick the better one. The RM learns to assign a scalar score to any (prompt, response) pair.

**3. PPO (Proximal Policy Optimization)** — the RL algorithm that updates the SFT model's weights to maximize the reward model's score, without drifting too far from the original SFT behavior (the "KL penalty" stops it from gaming the reward model).

```
prompt → SFT model → response → Reward Model → score
                                                  ↓
                               PPO updates SFT model weights to increase score
```

---

**Applied to RAG outcomes specifically**

RAG introduces a unique challenge: the model's response quality depends not just on the generation, but on whether it faithfully used the retrieved context. RL can be used to reinforce behaviors like:

- **Faithfulness** — did the answer come from the retrieved docs or did the model hallucinate?
- **Citation accuracy** — did it cite the right source?
- **Groundedness** — did it say "I don't know" when the context didn't contain the answer?

You'd design your reward signal to capture these:

```python
# Pseudo-code for a RAG reward function
def reward(prompt, retrieved_context, response):
    faithfulness_score  = check_claims_supported_by_context(response, retrieved_context)
    relevance_score     = check_response_answers_prompt(response, prompt)
    hallucination_score = penalise_unsupported_claims(response, retrieved_context)
    
    return faithfulness_score + relevance_score - hallucination_score
```

This is sometimes called **RLRF (RL from Retrieval Feedback)** in the research literature.

---

**Applied to OpenAI models specifically**

Here's where the practical reality bites — the same constraint as SFT applies. You **cannot run PPO against OpenAI's hosted models** because you need gradient access to update weights, and OpenAI doesn't expose that.

What you *can* do with OpenAI's API:

| Technique | Works via API? | What it does |
|---|---|---|
| RLHF / PPO | ❌ No | Needs weight access |
| DPO | ❌ No | Needs weight access |
| OpenAI Fine-Tuning API | ✅ Yes | SFT only (for now) |
| Prompt optimisation via RL | ✅ Yes | Treats model as black box |
| Best-of-N sampling | ✅ Yes | Poor man's RL |

---

**What you *can* do with OpenAI as a black box**

**Best-of-N sampling** is the simplest approximation — generate N responses, score them all with a reward function, return the best one:

```python
def best_of_n(prompt, context, n=5):
    responses = [call_openai(prompt, context) for _ in range(n)]
    scores    = [reward_function(r, context) for r in responses]
    return responses[scores.index(max(scores))]
```

**Prompt optimisation with RL** — treat the prompt itself as the thing being optimised. Use RL (or simpler methods like DSPy) to search over prompt variations and find the one that produces the highest-scoring outputs on average. The model weights never change, but the prompt does.

---

**The open-source path (where real RLHF lives)**

If you want true RL over a model you control, you'd use an open-source model + a library like **TRL (Transformer Reinforcement Learning)** from Hugging Face:

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# The value head is an extra scalar output added on top of the LM
# that PPO uses to estimate expected future reward
model  = AutoModelForCausalLMWithValueHead.from_pretrained("meta-llama/Llama-3-8B")
config = PPOConfig(learning_rate=1.41e-5, batch_size=16)
trainer = PPOTrainer(config, model, ref_model, tokenizer)

for prompt, response, reward in training_batches:
    trainer.step([prompt], [response], [reward])
```

---

**DPO — the modern alternative to PPO**

PPO is notoriously finicky to tune. Most teams in 2025 use **DPO (Direct Preference Optimization)** instead, which skips the separate reward model entirely and trains directly on preference pairs (chosen vs rejected responses). It's much more stable and gives similar results:

```python
# DPO training data format
{
  "prompt":   "What causes inflation?",
  "chosen":   "Inflation is caused by...",   # human preferred response
  "rejected": "Inflation just happens when..." # human dispreferred response
}
```

The math works out such that DPO implicitly optimises the same objective as RLHF but in a single supervised training step — no reward model, no PPO loop needed.

---

**Bottom line**

The honest practical answer is: if you're on OpenAI's API, real RL on the model weights isn't available to you today. Your options are best-of-N, prompt optimisation, or waiting to see if OpenAI exposes preference-based fine-tuning (they've hinted at it). If RL over model weights matters to your use case, that's the strongest argument for going open-source with Llama or Qwen, where you own the full pipeline from SFT through DPO/PPO.
