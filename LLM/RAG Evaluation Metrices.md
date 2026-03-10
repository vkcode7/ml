## RAG Evaluation Metrics

RAG has a **dual evaluation problem** — you must evaluate both the **retrieval** and the **generation** independently, then evaluate them **together**.

```
┌─────────────────────────────────────────────────────────┐
│              RAG EVALUATION FRAMEWORK                   │
│                                                         │
│  ┌─────────────────┐         ┌─────────────────────┐   │
│  │   RETRIEVAL     │ ──────► │    GENERATION       │   │
│  │   METRICS       │         │    METRICS          │   │
│  └─────────────────┘         └─────────────────────┘   │
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
