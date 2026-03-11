## RAG Evaluation Metrics

RAG has a **dual evaluation problem** — you must evaluate both the **retrieval** and the **generation** independently, then evaluate them **together**.

```
┌─────────────────────────────────────────────────────────┐
│              RAG EVALUATION FRAMEWORK                   │
│                                                         │
│  ┌─────────────────┐         ┌─────────────────────┐    │
│  │   RETRIEVAL     │ ──────► │    GENERATION       │    │
│  │   METRICS       │         │    METRICS          │    │
│  └─────────────────┘         └─────────────────────┘    │
│           │                           │                 │
│           └───────────┬───────────────┘                 │
│                       ▼                                 │
│              END-TO-END METRICS                         │
└─────────────────────────────────────────────────────────┘
```

---

### 1. Retrieval Metrics
*Did the system fetch the right chunks?*

| Metric | What it measures | Formula |
|---|---|---|
| **Precision@K** | Of K retrieved chunks, how many were relevant? | relevant_retrieved / K |
| **Recall@K** | Of all relevant chunks, how many were retrieved? | relevant_retrieved / total_relevant |
| **MRR (Mean Reciprocal Rank)** | How high was the first relevant chunk ranked? | avg(1 / rank_of_first_relevant) |
| **NDCG@K** | Ranking quality — rewards relevant chunks ranked higher | Normalized Discounted Cumulative Gain |
| **Context Relevance** | Are retrieved chunks relevant to the query? | LLM-as-judge score 0-1 |
| **Hit Rate** | Did at least one retrieved chunk contain the answer? | runs_with_hit / total_runs |

```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0

def mean_reciprocal_rank(retrieved_list: list[list], relevant_list: list[set]) -> float:
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example
retrieved = ["doc_3", "doc_1", "doc_5", "doc_2", "doc_4"]
relevant  = {"doc_1", "doc_2"}

print(precision_at_k(retrieved, relevant, k=3))  # 1/3 = 0.33
print(recall_at_k(retrieved, relevant, k=3))     # 1/2 = 0.50
```

---

### 2. Generation Metrics
*Was the generated answer good?*

#### 2a. Faithfulness (most critical for RAG)
*Is the answer grounded in the retrieved context — or hallucinated?*

```python
def evaluate_faithfulness(context: str, answer: str) -> dict:
    """LLM-as-judge faithfulness scorer."""
    prompt = f"""
    You are evaluating whether an AI answer is faithful to the context provided.

    CONTEXT:
    {context}

    ANSWER:
    {answer}

    Score faithfulness from 0.0 to 1.0 where:
    1.0 = every claim in the answer is supported by the context
    0.5 = some claims supported, some not
    0.0 = answer contradicts or ignores the context

    Respond ONLY in JSON:
    {{
        "faithfulness_score": 0.0-1.0,
        "unsupported_claims": ["list any claims not in context"],
        "reasoning": "brief explanation"
    }}
    """
    result = llm.invoke(prompt)
    return json.loads(result.content)
```

#### 2b. Answer Relevance
*Does the answer actually address the question asked?*

```python
def evaluate_answer_relevance(question: str, answer: str) -> dict:
    prompt = f"""
    Does this answer directly address the question?

    QUESTION: {question}
    ANSWER:   {answer}

    Respond ONLY in JSON:
    {{
        "relevance_score": 0.0-1.0,
        "addresses_question": true/false,
        "missing_aspects": ["aspects of question not addressed"],
        "reasoning": "brief explanation"
    }}
    """
    result = llm.invoke(prompt)
    return json.loads(result.content)
```

#### 2c. Other Generation Metrics

| Metric | What it measures | Method |
|---|---|---|
| **Faithfulness** | Answer grounded in context | LLM-as-judge |
| **Answer Relevance** | Answer addresses the question | LLM-as-judge |
| **Completeness** | All parts of question answered | LLM-as-judge |
| **Conciseness** | No unnecessary padding | LLM-as-judge |
| **BLEU** | N-gram overlap vs reference answer | Automated |
| **ROUGE-L** | Longest common subsequence vs reference | Automated |
| **BERTScore** | Semantic similarity vs reference | Embedding similarity |
| **Toxicity** | Harmful content in output | Classifier model |

```python
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# BERTScore — semantic similarity
def compute_bert_score(predictions: list, references: list) -> dict:
    P, R, F1 = bert_score(predictions, references, lang="en")
    return {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item()
    }

# ROUGE-L
def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }
```

---

### 3. End-to-End RAG Metrics
*How well does the full pipeline perform together?*

| Metric | What it measures |
|---|---|
| **Answer Correctness** | Final answer correct vs ground truth |
| **Context Utilization** | How much of retrieved context was actually used |
| **Hallucination Rate** | % of answers containing unsupported claims |
| **No-Answer Rate** | % of times model said "I don't know" correctly |
| **Noise Robustness** | Performance when irrelevant chunks are retrieved |
| **Counterfactual Robustness** | Performance when context contains wrong info |

```python
@dataclass
class RAGEvalResult:
    question:           str
    retrieved_chunks:   list
    generated_answer:   str
    reference_answer:   str

    # Retrieval
    precision_at_3:     float
    recall_at_3:        float
    mrr:                float
    context_relevance:  float

    # Generation
    faithfulness:       float
    answer_relevance:   float
    completeness:       float
    rouge_l:            float
    bert_score_f1:      float

    # End-to-end
    answer_correctness: float
    hallucinated:       bool

    @property
    def retrieval_score(self) -> float:
        return (
            self.precision_at_3    * 0.30 +
            self.recall_at_3       * 0.30 +
            self.mrr               * 0.20 +
            self.context_relevance * 0.20
        )

    @property
    def generation_score(self) -> float:
        return (
            self.faithfulness     * 0.35 +
            self.answer_relevance * 0.30 +
            self.completeness     * 0.20 +
            self.bert_score_f1    * 0.15
        )

    @property
    def overall_score(self) -> float:
        return (
            self.retrieval_score   * 0.40 +
            self.generation_score  * 0.60
        )

    @property
    def passed(self) -> bool:
        return (
            self.overall_score    >= 0.75 and
            self.faithfulness     >= 0.80 and  # hard floor
            not self.hallucinated
        )
```

---

### 4. RAGAS — Framework for RAG Evaluation

RAGAS is the de-facto open-source RAG evaluation library.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the client's current equity allocation?",
        "What are the risk limits for EM exposure?"
    ],
    "answer": [
        "The client currently holds 45% in equities.",
        "EM exposure is capped at 20% of total AUM."
    ],
    "contexts": [
        ["Client portfolio shows 45% equity, 35% fixed income, 20% alternatives."],
        ["Risk policy limits emerging market exposure to 20% of total portfolio value."]
    ],
    "ground_truth": [
        "Client equity allocation is 45%.",
        "EM exposure limit is 20% of AUM."
    ]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,           # answer grounded in context?
        answer_relevancy,       # answer relevant to question?
        context_precision,      # retrieved context precise?
        context_recall,         # retrieved context complete?
        answer_correctness,     # answer matches ground truth?
        answer_similarity       # semantic similarity to ground truth?
    ]
)

print(results)
# {
#   'faithfulness':        0.92,
#   'answer_relevancy':    0.88,
#   'context_precision':   0.85,
#   'context_recall':      0.79,
#   'answer_correctness':  0.84,
#   'answer_similarity':   0.91
# }
```

---

### 5. The RAG Failure Mode Matrix

| Failure | Retrieval | Generation | Metric that catches it |
|---|---|---|---|
| Wrong chunks fetched | ❌ | - | Precision@K, Context Relevance |
| Right chunks missed | ❌ | - | Recall@K, Hit Rate |
| Answer ignores context | - | ❌ | Faithfulness |
| Answer off-topic | - | ❌ | Answer Relevance |
| Hallucinated facts | - | ❌ | Faithfulness, Hallucination Rate |
| Incomplete answer | - | ❌ | Completeness, ROUGE |
| Noisy retrieval confuses model | ❌ | ❌ | Noise Robustness |
| Context window overflow | ❌ | ❌ | Context Utilization |

---

### 6. Thresholds & Targets

| Metric | Minimum | Target | Critical Floor |
|---|---|---|---|
| Faithfulness | 0.80 | > 0.90 | < 0.70 = block deploy |
| Answer Relevance | 0.75 | > 0.85 | < 0.65 = block deploy |
| Context Precision | 0.70 | > 0.85 | — |
| Context Recall | 0.70 | > 0.80 | — |
| Answer Correctness | 0.75 | > 0.85 | — |
| Hallucination Rate | < 5% | < 2% | > 10% = block deploy |
| Hit Rate | 0.85 | > 0.95 | — |
| BERTScore F1 | 0.75 | > 0.85 | — |

---

**One-liner summary:**
> "RAG evaluation is a pipeline problem — bad retrieval poisons generation, so you must score both independently with precision/recall/faithfulness metrics, then validate end-to-end with RAGAS or equivalent before you can trust the system in production."


# RAGAS Deep Dive

---

### What RAGAS Is

RAGAS (Retrieval Augmented Generation Assessment) is an **LLM-powered evaluation framework** — it uses LLMs to judge LLM outputs, removing the need for human-labeled ground truth on every test case.

```
┌─────────────────────────────────────────────────────────────┐
│                    RAGAS ARCHITECTURE                       │
│                                                             │
│  Your RAG Pipeline                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Query   │───►│Retriever │───►│Generator │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │                │                   │
│       ▼               ▼                ▼                   │
│  ┌─────────────────────────────────────────┐               │
│  │           RAGAS EVALUATOR               │               │
│  │                                         │               │
│  │  question + contexts + answer           │               │
│  │       + (optional) ground_truth         │               │
│  │                                         │               │
│  │  ┌──────────┐  ┌──────────┐             │               │
│  │  │Retrieval │  │Generation│             │               │
│  │  │ Metrics  │  │ Metrics  │             │               │
│  │  └──────────┘  └──────────┘             │               │
│  └─────────────────────────────────────────┘               │
│                       │                                     │
│                  Score 0.0–1.0                              │
└─────────────────────────────────────────────────────────────┘
```

---

### How Each Metric Works Internally

---

#### 1. Faithfulness
*Are all claims in the answer supported by the retrieved context?*

**Requires:** `question`, `answer`, `contexts`
**Does NOT require:** `ground_truth`

```
INTERNAL PROCESS:
─────────────────────────────────────────────────────
Step 1 — Statement Decomposition
  LLM breaks answer into atomic statements

  Answer: "The client holds 45% equities and
           risk limit is 20% for EM."

  Statements:
  [1] "Client holds 45% equities"
  [2] "Risk limit is 20% for EM"

Step 2 — Statement Verification
  For each statement, LLM checks:
  "Is this statement supported by the context?"
  → Yes / No per statement

Step 3 — Score Calculation
  faithfulness = supported_statements / total_statements
               = 2/2 = 1.0

  If statement [2] was NOT in context:
  faithfulness = 1/2 = 0.5
─────────────────────────────────────────────────────
```

```python
# What RAGAS does internally (simplified)
def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    combined_context = "\n".join(contexts)

    # Step 1: Decompose into statements
    decompose_prompt = f"""
    Break this answer into individual factual statements.
    Return as JSON list.

    Answer: {answer}
    """
    statements = json.loads(llm.invoke(decompose_prompt).content)

    # Step 2: Verify each statement
    supported = 0
    for statement in statements:
        verify_prompt = f"""
        Context: {combined_context}
        Statement: {statement}

        Is this statement fully supported by the context?
        Respond: {{"supported": true/false}}
        """
        result = json.loads(llm.invoke(verify_prompt).content)
        if result["supported"]:
            supported += 1

    # Step 3: Score
    return supported / len(statements) if statements else 0.0
```

---

#### 2. Answer Relevancy
*Does the answer directly address the question?*

**Requires:** `question`, `answer`, `contexts`
**Does NOT require:** `ground_truth`

```
INTERNAL PROCESS:
─────────────────────────────────────────────────────
Step 1 — Reverse Question Generation
  LLM generates N synthetic questions
  that the answer WOULD answer

  Answer: "The EM exposure limit is 20% of AUM"

  Generated questions:
  Q1: "What is the EM exposure limit?"
  Q2: "How much EM is allowed in the portfolio?"
  Q3: "What percentage cap exists for EM?"

Step 2 — Embedding Similarity
  Embed original question + each generated question
  Compute cosine similarity between original
  and each generated question

Step 3 — Score
  answer_relevancy = mean cosine similarity
                   = avg([sim(Q_orig, Q1),
                          sim(Q_orig, Q2),
                          sim(Q_orig, Q3)])

Key insight: If the answer is off-topic, the
generated questions won't resemble the original.
─────────────────────────────────────────────────────
```

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_answer_relevancy(question: str, answer: str, n: int = 3) -> float:

    # Step 1: Generate reverse questions
    prompt = f"""
    Generate {n} questions that this answer would correctly answer.
    Return as JSON list of strings.

    Answer: {answer}
    """
    generated_questions = json.loads(llm.invoke(prompt).content)

    # Step 2: Embed all questions
    original_embedding   = embeddings.embed_query(question)
    generated_embeddings = [embeddings.embed_query(q)
                            for q in generated_questions]

    # Step 3: Cosine similarities
    similarities = [
        cosine_similarity([original_embedding], [gen_emb])[0][0]
        for gen_emb in generated_embeddings
    ]

    return float(np.mean(similarities))
```

---

#### 3. Context Precision
*Are the retrieved chunks ranked well — relevant ones first?*

**Requires:** `question`, `contexts`, `ground_truth`

```
INTERNAL PROCESS:
─────────────────────────────────────────────────────
For each retrieved chunk at rank k:
  LLM judges: "Is this chunk relevant to answering
               the question given the ground truth?"
  → Yes / No

Then applies precision-weighted ranking:
  Rewards relevant chunks appearing EARLIER

Example — 4 retrieved chunks:
  Rank 1: Relevant   ✅
  Rank 2: Irrelevant ❌
  Rank 3: Relevant   ✅
  Rank 4: Irrelevant ❌

  Precision@1 = 1/1 = 1.0
  Precision@2 = 1/2 = 0.5
  Precision@3 = 2/3 = 0.67
  Precision@4 = 2/4 = 0.5

  context_precision = mean of precision@k
                      for relevant ranks only
                    = (1.0 + 0.67) / 2 = 0.835
─────────────────────────────────────────────────────
```

---

#### 4. Context Recall
*Did the retrieved chunks cover everything in the ground truth?*

**Requires:** `contexts`, `ground_truth`

```
INTERNAL PROCESS:
─────────────────────────────────────────────────────
Step 1 — Decompose ground truth into statements
  Ground truth: "EM limit is 20%. Client AUM is $10M.
                 Risk team approved this limit in 2024."

  Statements:
  [1] "EM limit is 20%"
  [2] "Client AUM is $10M"
  [3] "Risk team approved limit in 2024"

Step 2 — Attribute each statement to context
  LLM checks: "Can statement X be inferred
               from the retrieved context?"

  [1] → ✅ Found in context
  [2] → ✅ Found in context
  [3] → ❌ Not in any retrieved chunk

Step 3 — Score
  context_recall = attributed / total
                 = 2/3 = 0.67

  Low recall = retriever is missing important chunks
─────────────────────────────────────────────────────
```

---

#### 5. Answer Correctness
*Is the answer factually correct vs ground truth?*

**Requires:** `answer`, `ground_truth`

```
INTERNAL PROCESS:
─────────────────────────────────────────────────────
Combines TWO sub-scores:

1. Factual Similarity (LLM-as-judge)
   - Classifies each statement as:
     TP: in both answer and ground truth
     FP: in answer but not ground truth
     FN: in ground truth but not answer

   F1 = TP / (TP + 0.5*(FP+FN))

2. Semantic Similarity (BERTScore)
   - Embeds answer and ground truth
   - Computes cosine similarity

Final score:
  answer_correctness = (F1 * 0.75) + (semantic_sim * 0.25)
─────────────────────────────────────────────────────
```

---

### Full RAGAS Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from ragas.metrics.critique import harmfulness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
import pandas as pd

# ─────────────────────────────────────────────
# 1. CONFIGURE RAGAS TO USE YOUR LLM
# ─────────────────────────────────────────────
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Inject into metrics
for metric in [faithfulness, answer_relevancy,
               context_precision, context_recall, answer_correctness]:
    metric.llm        = ragas_llm
    metric.embeddings = ragas_embeddings

# ─────────────────────────────────────────────
# 2. PREPARE EVALUATION DATASET
# ─────────────────────────────────────────────
eval_data = {
    "question": [
        "What is the EM equity exposure limit for this client?",
        "What was the client's portfolio return last quarter?",
        "Who approved the risk policy change in Q4 2025?"
    ],
    "answer": [
        "The EM equity exposure limit is 20% of total AUM.",
        "The client portfolio returned 8.3% last quarter.",
        "The CRO approved the risk policy change in Q4 2025."
    ],
    "contexts": [
        # List of retrieved chunks per question
        [
            "Risk policy document: EM equity capped at 20% of AUM.",
            "Client mandate updated March 2026 to allow 20% EM."
        ],
        [
            "Q4 2025 report shows portfolio return of 8.3%.",
            "Benchmark return was 6.1% for same period."
        ],
        [
            "Policy amendment dated Nov 2025 signed by Head of Risk.",
            "Risk committee meeting minutes from Oct 2025."
        ]
    ],
    "ground_truth": [
        "EM equity exposure is limited to 20% of total AUM.",
        "Portfolio returned 8.3% in Q4 2025.",
        "The Head of Risk signed the policy amendment in November 2025."
    ]
}

dataset = Dataset.from_dict(eval_data)

# ─────────────────────────────────────────────
# 3. RUN EVALUATION
# ─────────────────────────────────────────────
run_config = RunConfig(
    timeout=60,
    max_retries=3,
    max_wait=30
)

results = evaluate(
    dataset    = dataset,
    metrics    = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ],
    run_config = run_config
)

# ─────────────────────────────────────────────
# 4. ANALYZE RESULTS
# ─────────────────────────────────────────────
df = results.to_pandas()

print("\n=== RAGAS EVALUATION RESULTS ===")
print(f"Faithfulness:       {results['faithfulness']:.3f}")
print(f"Answer Relevancy:   {results['answer_relevancy']:.3f}")
print(f"Context Precision:  {results['context_precision']:.3f}")
print(f"Context Recall:     {results['context_recall']:.3f}")
print(f"Answer Correctness: {results['answer_correctness']:.3f}")
print(f"Answer Similarity:  {results['answer_similarity']:.3f}")

# Per-question breakdown
print("\n=== PER QUESTION BREAKDOWN ===")
print(df[["question", "faithfulness", "answer_relevancy",
          "context_precision", "context_recall"]].to_string())

# Flag failures
print("\n=== FAILURES (faithfulness < 0.8) ===")
failures = df[df["faithfulness"] < 0.8]
for _, row in failures.iterrows():
    print(f"  Q: {row['question']}")
    print(f"  Faithfulness: {row['faithfulness']:.3f}")
    print(f"  Answer: {row['answer']}\n")
```

---

### Metric Dependency Map

```
Metric               question  answer  contexts  ground_truth
─────────────────────────────────────────────────────────────
Faithfulness            ✅        ✅       ✅          ❌
Answer Relevancy        ✅        ✅       ✅          ❌
Context Precision       ✅        ❌       ✅          ✅
Context Recall          ❌        ❌       ✅          ✅
Answer Correctness      ❌        ✅       ❌          ✅
Answer Similarity       ❌        ✅       ❌          ✅

✅ = required   ❌ = not needed
```

**Key insight:** Faithfulness and Answer Relevancy need **no ground truth** — useful for production monitoring where labeling every response is impractical.

---

### Best Practices

**1. Use GPT-4 class models as the judge**
```python
# ✅ DO this — stronger judge = more reliable scores
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# ❌ AVOID — weak judge produces unreliable scores
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
```

**2. Separate your judge LLM from your RAG LLM**
```python
# ✅ DO — use different model for evaluation
rag_llm   = ChatOpenAI(model="gpt-4o-mini")   # cheaper, faster for RAG
judge_llm = ChatOpenAI(model="gpt-4o")         # stronger for evaluation

# ❌ AVOID — judging your own outputs is biased
```

**3. Build a diverse, representative test set**
```python
# ✅ Cover all question types
test_set = {
    "simple_factual":    [...],   # "What is X?"
    "multi_hop":         [...],   # requires 2+ chunks
    "comparative":       [...],   # "What's the diff between X and Y?"
    "unanswerable":      [...],   # answer NOT in knowledge base
    "adversarial":       [...],   # edge cases, ambiguous queries
}

# ❌ AVOID — homogeneous test sets hide blind spots
```

**4. Always pass contexts as a list of chunks, not one blob**
```python
# ✅ Correct — preserves chunk-level precision scoring
"contexts": [
    ["chunk_1 text", "chunk_2 text", "chunk_3 text"]
]

# ❌ Wrong — merging chunks breaks context_precision ranking
"contexts": [
    ["chunk_1 text chunk_2 text chunk_3 text"]
]
```

**5. Run RAGAS in CI/CD as a regression gate**
```python
def ragas_regression_gate(results: dict, thresholds: dict) -> bool:
    failures = []
    for metric, threshold in thresholds.items():
        if results[metric] < threshold:
            failures.append(f"{metric}: {results[metric]:.3f} < {threshold}")

    if failures:
        print("❌ RAGAS GATE FAILED:")
        for f in failures:
            print(f"   {f}")
        return False

    print("✅ RAGAS GATE PASSED")
    return True

thresholds = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_precision": 0.70,
    "context_recall":    0.70,
    "answer_correctness":0.75
}

gate_passed = ragas_regression_gate(results, thresholds)
if not gate_passed:
    raise Exception("RAGAS regression gate failed — block deployment")
```

---

### Pitfalls to Avoid

| Pitfall | Problem | Fix |
|---|---|---|
| **Using weak judge LLM** | GPT-3.5 as judge gives noisy, unreliable scores | Always use GPT-4o or equivalent as judge |
| **Same LLM judges itself** | Self-evaluation bias inflates scores | Use separate, stronger model as judge |
| **Tiny test set** | 10-20 samples have high variance, misleading averages | Minimum 100 samples, 500+ for production |
| **Homogeneous questions** | Misses failure modes on edge cases | Cover simple, multi-hop, unanswerable, adversarial |
| **Merging chunks into one context** | Breaks context_precision ranking logic | Always pass contexts as list of individual chunks |
| **Ignoring unanswerable questions** | Model may hallucinate when answer isn't in KB | Include "unanswerable" test cases explicitly |
| **Running RAGAS once at launch** | Model, data, and queries drift over time | Run continuously in production monitoring |
| **Treating scores as absolute truth** | RAGAS is LLM-based — it can also be wrong | Spot-check low-scoring cases manually |
| **No baseline to compare against** | Scores mean nothing without reference | Always compare to a baseline run |
| **Ignoring per-question breakdown** | Aggregate scores hide per-question failures | Always analyze df per row, not just averages |

---

### RAGAS for Production Monitoring

```python
class RAGASProductionMonitor:
    """Run lightweight RAGAS eval continuously in production."""

    def __init__(self, sample_rate: float = 0.10):
        self.sample_rate = sample_rate   # evaluate 10% of live traffic
        self.results_log = []

    def should_evaluate(self) -> bool:
        import random
        return random.random() < self.sample_rate

    def evaluate_live_request(self, question: str, contexts: list,
                               answer: str) -> dict:
        """Run faithfulness + relevancy only — no ground truth needed."""
        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts]
        })

        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        record = {
            "timestamp":        datetime.utcnow().isoformat(),
            "question":         question,
            "faithfulness":     results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "alert":            results["faithfulness"] < 0.70
        }

        self.results_log.append(record)

        if record["alert"]:
            print(f"🚨 Low faithfulness alert: {results['faithfulness']:.3f}")
            print(f"   Q: {question}")

        return record
```

---

### When to Use Which Metrics

| Scenario | Use these metrics |
|---|---|
| **Development & testing** | All 6 metrics with ground truth |
| **Production monitoring** | Faithfulness + Answer Relevancy (no ground truth needed) |
| **Retriever tuning** | Context Precision + Context Recall |
| **Generator tuning** | Faithfulness + Answer Correctness |
| **CI/CD regression gate** | All metrics with hard thresholds |
| **Debugging hallucinations** | Faithfulness (with statement-level breakdown) |

---

**One-liner summary:**
> "RAGAS works by using LLMs to decompose, verify, and score each component of your RAG pipeline — the key is using a strong, independent judge model, building a diverse test set that includes unanswerable questions, and running it continuously rather than just at launch."
