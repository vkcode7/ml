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
