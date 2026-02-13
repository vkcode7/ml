Below is a **clean end-to-end example** of how you would LoRA-fine-tune an OpenAI model in Python using a corpus of ~10,000 SEC 10-K documents from U.S. Securities and Exchange Commission (SEC.gov).

> ⚠️ Important reality:
> OpenAI’s hosted fine-tuning abstracts the underlying PEFT method (e.g., LoRA). You don’t manually implement LoRA layers — the platform handles parameter-efficient fine-tuning internally.
> So you fine-tune via API; LoRA happens under the hood.

---

# 🎯 Goal

Fine-tune a model so it:

* Understands financial reporting language
* Answers questions grounded in 10-K filings
* Summarizes risk factors, MD&A, etc.
* Produces structured financial analysis

---

# 🧠 Step 0 — Decide the Strategy

You generally **don’t fine-tune directly on raw 10-K text**.

Better pipeline:

```
Raw 10-K docs
   ↓
Chunk
   ↓
Create instruction/QA pairs
   ↓
SFT fine-tuning
```

Why?

Fine-tuning teaches:

* Style
* Task behavior
* Structured reasoning

RAG handles:

* Factual grounding
* Specific filings

In production, you'd combine both.

---

# 🧱 Step 1 — Prepare Training Data

You need JSONL in chat format:

```json
{"messages":[
  {"role":"system","content":"You are a financial analyst."},
  {"role":"user","content":"Summarize the risk factors for Apple in 2023."},
  {"role":"assistant","content":"Apple's 2023 10-K highlights risks including supply chain dependence, regulatory scrutiny, foreign exchange volatility..."}
]}
```

---

## 🔹 Example Data Preparation Script

```python
import json
import glob

def create_training_examples():
    examples = []
    
    for file in glob.glob("sec_docs/*.txt"):
        with open(file, "r") as f:
            text = f.read()
        
        # Example transformation (simplified)
        prompt = "Summarize the key risk factors from this 10-K."
        response = summarize_risk_section(text)  # your custom parser
        
        example = {
            "messages": [
                {"role": "system", "content": "You are a CFA-level financial analyst."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        }
        
        examples.append(example)

    return examples


def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


training_data = create_training_examples()
save_jsonl(training_data, "financial_finetune.jsonl")
```

You’d generate thousands of such instruction-style examples.

---

# 🚀 Step 2 — Upload File to OpenAI

```python
from openai import OpenAI

client = OpenAI()

file = client.files.create(
    file=open("financial_finetune.jsonl", "rb"),
    purpose="fine-tune"
)

print(file.id)
```

---

# 🔥 Step 3 — Create Fine-Tuning Job

```python
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",  # example fine-tunable model
    hyperparameters={
        "n_epochs": 3
    }
)

print(job.id)
```

---

# 📊 Step 4 — Monitor Training

```python
status = client.fine_tuning.jobs.retrieve(job.id)
print(status.status)
```

When complete, you’ll receive:

```
fine_tuned_model: ft:gpt-4o-mini:your-org:financial-analyst:abc123
```

---

# 🧠 Step 5 — Use Your Fine-Tuned Model

```python
response = client.chat.completions.create(
    model="ft:gpt-4o-mini:your-org:financial-analyst:abc123",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": "Analyze Tesla's liquidity risks."}
    ]
)

print(response.choices[0].message.content)
```

---

# 📈 What LoRA Is Doing Behind the Scenes

Internally:

* Base model weights frozen
* Small low-rank matrices added
* Only small subset trained
* Memory-efficient updates

You don’t manually configure rank — OpenAI manages that.

---

# 🏗 Production Architecture (Recommended)

For financial use cases:

```
Fine-tuned model (style + reasoning)
         +
RAG (actual filings retrieval)
```

Flow:

```
User Question
   ↓
Vector search over 10-K chunks
   ↓
Inject relevant sections
   ↓
Fine-tuned model response
```

This prevents hallucination.

---

# ⚖ When Fine-Tuning Is Worth It

Use fine-tuning if you need:

* Consistent financial tone
* Structured outputs (e.g., risk tables)
* Domain-specific reasoning style
* Regulatory-safe language

Do NOT fine-tune just to “store” 10-K facts.

That’s what RAG is for.

---

# 🧠 Practical Advice (Finance Domain)

For 10,000 SEC docs:

1. Build RAG first
2. Then fine-tune on:

   * Structured financial analysis examples
   * Risk summarization
   * Earnings call Q&A
   * Ratio interpretation

This gives best ROI.

---

# 💰 Cost Reality

Fine-tuning cost depends on:

* Tokens in dataset
* Number of epochs
* Model size

10,000 filings → likely millions of tokens
You may want:

* Section-level sampling
* Not full-document training

---

# 🎯 End-to-End Summary

```
SEC 10-K corpus
   ↓
Chunk + convert to instruction pairs
   ↓
Upload JSONL
   ↓
Create fine-tuning job
   ↓
Deploy fine-tuned model
   ↓
Combine with RAG in production
```

---

If you'd like, I can next:

* Show a RAG + fine-tuned hybrid architecture
* Estimate token cost for 10,000 documents
* Show C# version of deployment (since you use .NET)


# Show how to auto-generate training pairs from filings

Below is a **practical, production-style pipeline** to auto-generate instruction/response training pairs from 10-K filings obtained from U.S. Securities and Exchange Commission.

We’ll turn raw filings into high-quality SFT training data.

---

# 🎯 Goal

Convert:

```
Raw 10-K text (10000 docs)
```

Into:

```
Instruction → High-quality assistant answer
```



For fine-tuning.

---

# 🧠 High-Level Strategy

We do NOT train on raw filings.

We:

1. Parse sections
2. Chunk intelligently
3. Generate synthetic Q&A pairs
4. Filter for quality
5. Export JSONL for fine-tuning

---

# 🧱 Step 1 — Parse Structured Sections

10-Ks follow predictable structure:

* Item 1A – Risk Factors
* Item 7 – MD&A
* Item 8 – Financial Statements
* etc.

We extract by regex or XML parsing.

```python
import re

def extract_section(text, section_title):
    pattern = rf"{section_title}(.+?)(Item\s+\d+[A-Z]?)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None
```

Example:

```python
risk_section = extract_section(doc_text, "Item 1A")
mda_section = extract_section(doc_text, "Item 7")
```

---

# 🧩 Step 2 — Chunk the Text

Chunk size should be:

* 800–1500 tokens
* Section-aware
* Sentence-boundary aligned

```python
from nltk.tokenize import sent_tokenize

def chunk_text(text, max_chars=4000):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks
```

---

# 🧠 Step 3 — Auto-Generate Training Pairs

We now use a strong base model to generate Q&A pairs.

Prompt template:

```python
GEN_PROMPT = """
You are a CFA-level financial analyst.

From the following 10-K excerpt, generate:
1. One analytical question an investor might ask.
2. A detailed, professional answer grounded only in the text.
3. A short structured summary (bullet points).

Text:
{chunk}
"""
```

---

## Python Generation Code

```python
from openai import OpenAI

client = OpenAI()

def generate_pair(chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": GEN_PROMPT.format(chunk=chunk)}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
```

---

# 🧾 Step 4 — Convert to Fine-Tune Format

We structure it as:

```python
import json

def format_for_sft(question, answer):
    return {
        "messages": [
            {"role": "system", "content": "You are a CFA-level financial analyst."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
```

You may need to parse the generated text into Q/A fields.

---

# 🧪 Step 5 — Quality Filtering (Critical)

We should filter:

* Very short answers
* Generic responses
* Hallucinated content
* Repetitive samples

Simple heuristics:

```python
def quality_filter(answer):
    if len(answer) < 300:
        return False
    if "not mentioned" in answer.lower():
        return False
    return True
```

Advanced filtering:

* Run a second model pass to check grounding
* Check if answer quotes text
* Use semantic similarity vs source chunk

---

# 🏗 Step 6 — Pipeline Over 10,000 Documents

```python
import glob

training_data = []

for file in glob.glob("sec_docs/*.txt"):
    with open(file, "r") as f:
        text = f.read()

    risk_section = extract_section(text, "Item 1A")
    if not risk_section:
        continue

    chunks = chunk_text(risk_section)

    for chunk in chunks:
        generated = generate_pair(chunk)
        question, answer = parse_generated(generated)

        if quality_filter(answer):
            training_data.append(format_for_sft(question, answer))

save_jsonl(training_data, "financial_training.jsonl")
```

---

# 📊 Expected Yield

From 10,000 filings:

* ~5 sections per filing
* ~5 chunks per section
* ~1 QA per chunk

→ Potentially 200k+ training examples

You may downsample to 20–50k high-quality examples.

---

# 🧠 What This Achieves

You are teaching the model:

* Financial reasoning style
* Risk analysis patterns
* Professional tone
* Structured outputs
* SEC-style language

NOT memorization.

---

# ⚠️ Important: Avoid Data Leakage

Do NOT include:

* Exact company-specific facts in evaluation
* Future-dated filings
* Sensitive proprietary overlays

---

# 🔥 Advanced Improvements

### 1️⃣ Generate Multiple Question Types

For each chunk:

* Risk analysis question
* Ratio interpretation
* Liquidity analysis
* Forward-looking uncertainty

---

### 2️⃣ Add Structured Output Examples

Train on outputs like:

```
Liquidity Risk:
Operational Risk:
Regulatory Risk:
Market Risk:
```

---

### 3️⃣ Add Negative Examples (for DPO later)

Generate:

* Weak answer
* Strong answer
* Label preferred

Useful for alignment training.

---

# 🎯 Best Production Strategy

Do BOTH:

* Fine-tune for style and reasoning
* Use RAG for factual grounding

Because:

Fine-tuning ≠ factual database

---

# 🧠 Final Mental Model

Raw filings → Structured chunks → Synthetic analyst Q&A → Filter → JSONL → Fine-tune

You’re teaching the model:

> “How to think like a financial analyst”

---

If you'd like, I can next:

* Show how to estimate token cost for 10,000 filings
* Show how to combine this with RAG
* Show how to detect hallucinations during generation


# Show evaluation metrics for financial fine-tuning

Excellent — this is where most fine-tuning projects fail.

For financial 10-K fine-tuning, evaluation must measure:

1. **Financial correctness**
2. **Grounding to filings**
3. **Analytical quality**
4. **Hallucination risk**
5. **Style consistency**
6. **General capability retention**

Below is a production-grade evaluation framework.

---

# 🎯 1️⃣ Core Evaluation Categories

| Category           | What It Measures                 | Why It Matters            |
| ------------------ | -------------------------------- | ------------------------- |
| Factual Accuracy   | Correct financial interpretation | Avoid regulatory risk     |
| Groundedness       | Answer supported by source text  | Prevent hallucinations    |
| Analytical Depth   | True financial reasoning         | Not surface summarization |
| Consistency        | Stable structured output         | Enterprise usability      |
| General Capability | Model didn’t degrade             | Avoid overfitting         |

---

# 🧠 2️⃣ Offline Automatic Metrics

These are scalable.

---

## 🔹 A. Exactness / Overlap (ROUGE / BLEU)

Useful only for:

* Summarization tasks
* Structured templates

But limited for reasoning.

Use lightly.

---

## 🔹 B. Semantic Similarity (Embedding-Based)

Compare generated answer to:

* Gold reference answer
* Source chunk

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Metric:

```
sim(generated, reference)
```

Better than ROUGE for finance language.

---

## 🔹 C. Groundedness Score (Critical)

We check:

> Does every major claim appear in source text?

Prompt a model:

```text
Given the source text and the model answer,
identify unsupported claims.
Return a hallucination score from 0–1.
```

Metric:

```
Hallucination Rate = (# unsupported claims) / (total claims)
```

This is extremely important for SEC data.

---

# 🏦 3️⃣ Financial Reasoning Evaluation

We must test real financial logic.

Create evaluation tasks like:

* “Explain how interest rate increases impact this company’s debt profile.”
* “Assess liquidity risk using balance sheet excerpts.”
* “Compare current year vs prior year revenue drivers.”

Have:

* Gold expert-written answers
* Or rubric-based scoring

---

## 🔹 Rubric Scoring (Very Effective)

Define scoring dimensions:

| Dimension              | Score 1–5 |
| ---------------------- | --------- |
| Correct interpretation |           |
| Risk identification    |           |
| Causal reasoning       |           |
| Financial terminology  |           |
| Regulatory awareness   |           |

You can automate rubric scoring via LLM evaluator.

Example evaluator prompt:

```text
Score the answer from 1-5 for:
- Financial correctness
- Depth of analysis
- Use of financial terminology
Return JSON only.
```

---

# 🔬 4️⃣ Hallucination Detection Metrics

### 🔹 Unsupported Fact Rate

Claims not present in filing.

### 🔹 Fabricated Numeric Rate

Invented numbers not in source.

Extremely important for financial systems.

---

# 📊 5️⃣ Regression Testing (Before vs After FT)

Always compare:

| Metric                      | Base Model | Fine-Tuned |
| --------------------------- | ---------- | ---------- |
| Financial QA accuracy       |            |            |
| Hallucination rate          |            |            |
| Structured format adherence |            |            |
| General knowledge QA        |            |            |

We want:

* Domain improvement
* No general collapse

---

# 🧪 6️⃣ Hold-Out Filing Evaluation

Never evaluate on training filings.

Use:

* Different companies
* Different years
* Different industries

Test:

* Cross-sector generalization
* New regulatory language

---

# 🧠 7️⃣ Style & Formatting Consistency

Check:

* Structured sections present
* Bullet formatting correct
* Risk categories consistently labeled

Metric example:

```
Format Adherence Rate =
(# responses following template) / (total responses)
```

---

# 📈 8️⃣ RAG-Aware Evaluation

If using RAG:

Measure:

* Retrieval accuracy (is correct section retrieved?)
* Answer grounded in retrieved text?
* Context utilization rate

Metric:

```
Context Utilization Score =
% of answer sentences that cite retrieved chunk
```

---

# 🧠 9️⃣ Catastrophic Forgetting Check

Test general tasks:

* Basic math
* Common reasoning
* Non-financial QA

We want no degradation.

---

# 🔥 10️⃣ Human Expert Evaluation (Gold Standard)

For financial applications:

Have CFA / CPA review sample outputs.

Score:

* Regulatory safety
* Investment-grade reasoning
* Risk nuance
* Disclosure awareness

Even 100 human-reviewed samples is powerful.

---

# 📊 Example Full Evaluation Dashboard

| Metric                       | Target   |
| ---------------------------- | -------- |
| Financial QA accuracy        | >85%     |
| Hallucination rate           | <5%      |
| Unsupported numeric claims   | 0%       |
| Format adherence             | >95%     |
| Cross-sector generalization  | Stable   |
| General knowledge regression | <2% drop |

---

# 🎯 What Matters Most in Finance

1. Hallucination rate
2. Fabricated numbers
3. Misinterpretation of risk
4. Overconfident tone

These are regulatory liabilities.

---

# 🧠 Final Mental Model

Financial fine-tuning evaluation is not:

> “Does it sound smart?”

It is:

> “Is it grounded, correct, structured, and safe?”

---

If you'd like, I can next:

* Show an automated evaluation pipeline in Python
* Show how to build a financial LLM benchmark dataset
* Show how to quantify hallucination mathematically
* Show how to detect numeric fabrication
* Show a full evaluation workflow integrated with CI/CD

# Esimate Token cost for 10K documents

Let’s estimate token cost for **10,000 SEC filings (10-K)** for fine-tuning and/or inference.

We’ll assume:

* Average 10-K length: **70,000–120,000 words**
* Rough rule: **1 token ≈ 0.75 words**
* So: **1 word ≈ 1.3 tokens**

---

# 1️⃣ Estimate Tokens Per Document

### Conservative estimate

* 80,000 words per filing
* Tokens ≈ 80,000 × 1.3
* ≈ **104,000 tokens per document**

### Total for 10,000 documents

```
104,000 × 10,000 = 1,040,000,000 tokens
```

👉 **~1.0–1.2 billion tokens total**

That’s realistic for full 10-Ks.

---

# 2️⃣ But You Rarely Train on Full Raw Documents

In practice you:

* Chunk into 1–4k token segments
* Extract Q&A pairs
* Filter boilerplate
* Remove tables / exhibits

Typical reduction:

| Strategy                  | Effective Tokens |
| ------------------------- | ---------------- |
| Raw full filings          | ~1B tokens       |
| Remove boilerplate        | ~600–800M        |
| Q/A extraction only       | ~100–300M        |
| Targeted risk & MD&A only | ~150–400M        |

---

# 3️⃣ Training Cost Estimate (Example)

Let’s assume:

* 300M tokens after cleaning
* 3 training epochs
* Effective tokens seen = 900M tokens

Now assume hypothetical fine-tuning price:

```
$X per 1M tokens
```

If cost = $8 per 1M tokens (example pricing class):

```
900M / 1M × $8
= 900 × 8
= $7,200
```

If cost = $4 per 1M:

```
= $3,600
```

If cost = $2 per 1M:

```
= $1,800
```

So realistic range:

👉 **$2K – $10K** depending on pricing tier and preprocessing.

---

# 4️⃣ Inference Cost (Post Fine-Tuning)

Assume:

* Avg prompt: 2,000 tokens
* Avg output: 500 tokens
* 10,000 queries per month

Tokens per query:

```
2,500 tokens
```

Monthly tokens:

```
2,500 × 10,000 = 25M tokens
```

Inference cost depends on:

* Model size
* Input vs output pricing

But expect **hundreds to low thousands per month** at scale.

---

# 5️⃣ Important Real-World Consideration

Training on **1B tokens is massive**.

Most financial fine-tuning setups:

* Use 50M–300M tokens
* Or use RAG instead of full fine-tuning
* Or LoRA adapters on open weights

For SEC filings specifically:

👉 RAG + small instruction tuning often beats brute-force full corpus training.

---

# 6️⃣ Quick Mental Model

| Corpus Size    | Tokens       | Scale      |
| -------------- | ------------ | ---------- |
| 1 filing       | ~100k tokens | small      |
| 100 filings    | ~10M tokens  | medium     |
| 10,000 filings | ~1B tokens   | very large |

---

If you'd like, I can next:

* Show Python code to compute exact token counts using `tiktoken`
* Estimate cost for specific OpenAI model pricing
* Compare cost of RAG vs fine-tuning
* Show optimal chunking strategy for 10-Ks


