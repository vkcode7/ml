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

* Estimate cost for specific OpenAI model pricing
* Compare cost of RAG vs fine-tuning
* Show optimal chunking strategy for 10-Ks


# Show Python code to compute exact token counts using `tiktoken`

Here's how to count tokens using tiktoken, which is the official tokenizer library from OpenAI:

```py
import tiktoken

# Basic token counting
text = "Hello! How are you doing today?"

# For GPT-4, GPT-3.5, and other OpenAI models
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"Decoded tokens: {[encoding.decode([token]) for token in tokens]}")
```

**Output:**
```
Text: Hello! How are you doing today?
Token count: 7
Tokens: [9906, 0, 2650, 527, 499, 3815, 3432, 30]
Decoded tokens: ['Hello', '!', ' How', ' are', ' you', ' doing', ' today', '?']
```

Different Models Use Different Encodings

```python
import tiktoken

text = "The quick brown fox jumps over the lazy dog."

# Different encodings for different models
encodings = {
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "text-davinci-003": tiktoken.encoding_for_model("text-davinci-003"),
}

# Or use encoding names directly
# cl100k_base: GPT-4, GPT-3.5-turbo
# p50k_base: Codex models, text-davinci-002/003
# r50k_base: GPT-3 models like davinci

for model_name, encoding in encodings.items():
    token_count = len(encoding.encode(text))
    print(f"{model_name}: {token_count} tokens")
```

Truncating Text to Fit Token Limits
```python
import tiktoken

def truncate_to_token_limit(text, max_tokens, model="gpt-4"):
    """Truncate text to fit within a token limit."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

long_text = "This is a very long text. " * 100
max_tokens = 50

truncated = truncate_to_token_limit(long_text, max_tokens)
print(f"Original tokens: {len(tiktoken.encoding_for_model('gpt-4').encode(long_text))}")
print(f"Truncated tokens: {len(tiktoken.encoding_for_model('gpt-4').encode(truncated))}")
print(f"Truncated text: {truncated}...")
```

# A 10-K Chunking Strategy

# Optimal Chunking Strategy for 10-K SEC Filings

10-Ks are tricky because they're long (50-300 pages), have specific sections, tables, and legal language. Here's a production-ready chunking strategy:

```python
import tiktoken
from typing import List, Dict
import re

class TenKChunker:
    def __init__(self, model="gpt-4", chunk_size=1000, overlap=200):
        """
        chunk_size: target tokens per chunk (1000-1500 is good for 10-Ks)
        overlap: tokens to overlap between chunks (helps maintain context)
        """
        self.encoding = tiktoken.encoding_for_model(model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract major 10-K sections. These are standardized by SEC.
        """
        sections = {
            "Item 1": "Business",
            "Item 1A": "Risk Factors", 
            "Item 1B": "Unresolved Staff Comments",
            "Item 2": "Properties",
            "Item 3": "Legal Proceedings",
            "Item 4": "Mine Safety Disclosures",
            "Item 5": "Market for Registrant's Common Equity",
            "Item 6": "Selected Financial Data",
            "Item 7": "MD&A",  # Management Discussion & Analysis - VERY IMPORTANT
            "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
            "Item 8": "Financial Statements",
            "Item 9": "Changes in and Disagreements with Accountants",
            "Item 9A": "Controls and Procedures",
            "Item 9B": "Other Information",
            "Item 10": "Directors, Executive Officers and Corporate Governance",
            "Item 11": "Executive Compensation",
            "Item 12": "Security Ownership",
            "Item 13": "Certain Relationships and Related Transactions",
            "Item 14": "Principal Accounting Fees and Services",
            "Item 15": "Exhibits, Financial Statement Schedules"
        }
        
        extracted = {}
        
        # Pattern to match item headers (case insensitive, flexible formatting)
        for item_num, item_name in sections.items():
            # Match patterns like "Item 1." or "ITEM 1 -" or "Item 1:"
            pattern = rf"{item_num}[\.\:\-\s]+{item_name}"
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                start_pos = match.start()
                
                # Find the next item to determine end position
                next_item_pos = len(text)
                for next_item in sections.keys():
                    if next_item > item_num:  # Only look for items that come after
                        next_pattern = rf"{next_item}[\.\:\-\s]+"
                        next_match = re.search(next_pattern, text[start_pos+100:], re.IGNORECASE)
                        if next_match:
                            next_item_pos = min(next_item_pos, start_pos + 100 + next_match.start())
                
                extracted[item_num] = text[start_pos:next_item_pos].strip()
        
        return extracted
    
    def chunk_with_semantic_boundaries(self, text: str, section_name: str = "") -> List[Dict]:
        """
        Chunk text with awareness of semantic boundaries (paragraphs, sentences).
        Better than hard token cutoffs.
        """
        chunks = []
        
        # Split by paragraphs first (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = self.count_tokens(para)
            
            # If single paragraph exceeds chunk size, split by sentences
            if para_tokens > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'section': section_name
                    })
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'tokens': current_tokens,
                            'section': section_name
                        })
                        # Keep overlap from end of previous chunk
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + " " + sent
                        current_tokens = self.count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sent if current_chunk else sent
                        current_tokens += sent_tokens
            
            # Normal case: add paragraph to current chunk
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens = self.count_tokens(current_chunk)
            
            # Paragraph would exceed limit, start new chunk
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': current_tokens,
                        'section': section_name
                    })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                current_tokens = self.count_tokens(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'section': section_name
            })
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get the last N tokens for overlap."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.overlap:
            return text
        
        overlap_tokens = tokens[-self.overlap:]
        return self.encoding.decode(overlap_tokens)
    
    def chunk_10k(self, text: str, preserve_sections: bool = True) -> List[Dict]:
        """
        Main method: chunk a 10-K filing optimally.
        
        preserve_sections=True: Keep section boundaries intact (recommended)
        preserve_sections=False: Treat as one document
        """
        if preserve_sections:
            # Extract sections first
            sections = self.extract_sections(text)
            
            all_chunks = []
            
            # Chunk each section separately
            for section_id, section_text in sections.items():
                section_chunks = self.chunk_with_semantic_boundaries(
                    section_text, 
                    section_name=section_id
                )
                all_chunks.extend(section_chunks)
            
            # If some text wasn't captured in sections, chunk it too
            total_section_length = sum(len(s) for s in sections.values())
            if total_section_length < len(text) * 0.8:  # Less than 80% captured
                remaining = self.chunk_with_semantic_boundaries(text, "Uncategorized")
                all_chunks.extend(remaining)
            
            return all_chunks
        else:
            return self.chunk_with_semantic_boundaries(text)


# Example usage
if __name__ == "__main__":
    # Simulated 10-K excerpt
    sample_10k = """
    Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
    
    The following discussion should be read in conjunction with our consolidated financial statements.
    
    Overview
    
    Our company operates in three segments: Cloud Services, Software Licensing, and Hardware Systems.
    Revenue increased 15% year-over-year driven primarily by cloud adoption.
    
    We continue to invest heavily in AI and machine learning capabilities to enhance our product offerings.
    Operating margins improved from 23% to 26% due to operational efficiencies and economies of scale.
    
    Critical Accounting Policies
    
    Revenue Recognition: We recognize revenue when control of promised goods or services transfers to customers.
    For multi-year cloud contracts, we recognize revenue ratably over the contract term.
    
    Stock-Based Compensation: We measure stock-based awards at grant-date fair value and recognize 
    expense over the vesting period.
    
    Item 7A. Quantitative and Qualitative Disclosures About Market Risk
    
    We are exposed to market risks including interest rate risk and foreign currency risk.
    Our international operations account for 45% of total revenue, exposing us to currency fluctuations.
    """ * 50  # Simulate longer document
    
    # Initialize chunker
    chunker = TenKChunker(
        model="gpt-4",
        chunk_size=1000,  # ~1000 tokens per chunk
        overlap=200       # 200 token overlap
    )
    
    # Chunk the 10-K
    chunks = chunker.chunk_10k(sample_10k, preserve_sections=True)
    
    # Display results
    print(f"Total chunks created: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
        print(f"Chunk {i}:")
        print(f"  Section: {chunk['section']}")
        print(f"  Tokens: {chunk['tokens']}")
        print(f"  Preview: {chunk['text'][:200]}...")
        print()
```

---

## **Advanced: Table-Aware Chunking**

10-Ks have lots of tables. Tables break badly with naive chunking:

```python
import re
from typing import List, Dict

class TableAwareChunker(TenKChunker):
    def detect_tables(self, text: str) -> List[Dict]:
        """
        Detect table-like structures in text.
        Tables often have multiple columns aligned with spaces/tabs.
        """
        tables = []
        lines = text.split('\n')
        
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            # Heuristic: tables have multiple whitespace-separated columns
            # and consistent alignment
            columns = len(re.findall(r'\s{2,}', line))
            
            if columns >= 3:  # At least 3 columns
                if not in_table:
                    in_table = True
                    table_start = i
                table_lines.append(line)
            else:
                if in_table and len(table_lines) >= 3:  # Minimum 3 rows
                    tables.append({
                        'start_line': table_start,
                        'end_line': i - 1,
                        'content': '\n'.join(table_lines)
                    })
                in_table = False
                table_lines = []
        
        return tables
    
    def chunk_with_table_preservation(self, text: str, section_name: str = "") -> List[Dict]:
        """
        Keep tables intact in single chunks when possible.
        """
        tables = self.detect_tables(text)
        chunks = []
        
        # Split text around tables
        current_pos = 0
        
        for table in tables:
            # Chunk text before table
            before_table = text[current_pos:text.find(table['content'])]
            if before_table.strip():
                chunks.extend(
                    self.chunk_with_semantic_boundaries(before_table, section_name)
                )
            
            # Add table as its own chunk (or split if too large)
            table_tokens = self.count_tokens(table['content'])
            if table_tokens <= self.chunk_size * 1.5:  # Allow tables to be slightly larger
                chunks.append({
                    'text': table['content'],
                    'tokens': table_tokens,
                    'section': section_name,
                    'type': 'table'
                })
            else:
                # Table too large, split by rows
                table_rows = table['content'].split('\n')
                header = table_rows[0] if table_rows else ""
                
                current_table_chunk = header
                current_tokens = self.count_tokens(header)
                
                for row in table_rows[1:]:
                    row_tokens = self.count_tokens(row)
                    if current_tokens + row_tokens > self.chunk_size:
                        chunks.append({
                            'text': current_table_chunk,
                            'tokens': current_tokens,
                            'section': section_name,
                            'type': 'table_fragment'
                        })
                        current_table_chunk = header + "\n" + row
                        current_tokens = self.count_tokens(current_table_chunk)
                    else:
                        current_table_chunk += "\n" + row
                        current_tokens += row_tokens
                
                if current_table_chunk:
                    chunks.append({
                        'text': current_table_chunk,
                        'tokens': current_tokens,
                        'section': section_name,
                        'type': 'table_fragment'
                    })
            
            current_pos = text.find(table['content']) + len(table['content'])
        
        # Chunk remaining text after last table
        remaining = text[current_pos:]
        if remaining.strip():
            chunks.extend(
                self.chunk_with_semantic_boundaries(remaining, section_name)
            )
        
        return chunks
```

---

## **Production Tips for 10-K Processing**

```python
# Recommended settings for different use cases

# For Q&A / RAG systems
chunker = TenKChunker(
    chunk_size=1000,    # Good balance for retrieval
    overlap=200         # Ensures context isn't lost
)

# For summarization
chunker = TenKChunker(
    chunk_size=2000,    # Larger chunks for better context
    overlap=100         # Less overlap needed
)

# For fine-tuning data
chunker = TenKChunker(
    chunk_size=512,     # Match your model's context window
    overlap=0           # No overlap for training data
)
```

**Key Insights:**
- **Always preserve section boundaries** - Item 7 (MD&A) is gold for analysis
- **Use overlap** - 10-Ks reference prior sections constantly
- **Handle tables specially** - They contain critical financial data
- **Metadata matters** - Tag chunks with section, year, company ticker
- **Token counts vary** - Financial jargon tokenizes differently than normal text

This approach gives you semantically meaningful chunks that LLMs can actually understand and reason about!
