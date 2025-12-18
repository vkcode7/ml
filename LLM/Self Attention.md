
# The Process: From Raw Scores to Percentages
## Step 1: Create Query, Key, Value (Using Learned Weights)

Each word starts as an embedding (a vector of numbers). Let's say each word is represented by a simple vector:
```
"Cat"   = [0.2, 0.8, 0.3]
"ate"   = [0.5, 0.4, 0.9]
"mouse" = [0.1, 0.7, 0.6]
```
Now, the model has learned weight matrices called W_Q, W_K, W_V (these are learned during training, just like any other neural network weights).
We multiply each word embedding by these matrices:
```
Query("ate")  = "ate" × W_Q  →  [0.3, 0.6]
Key("Cat")    = "Cat" × W_K  →  [0.8, 0.2]
Key("ate")    = "ate" × W_K  →  [0.4, 0.7]
Key("mouse")  = "mouse" × W_K → [0.9, 0.1]
```
## Step 2: Calculate Raw Attention Scores (Dot Product)
Now we compare the Query of "ate" with the Key of each word using the dot product:
```
The dot product measures similarity between two vectors.
Score("ate" → "Cat") = Query("ate") · Key("Cat")
                     = [0.3, 0.6] · [0.8, 0.2]
                     = (0.3 × 0.8) + (0.6 × 0.2)
                     = 0.24 + 0.12
                     = 0.36

Score("ate" → "ate") = Query("ate") · Key("ate")
                     = [0.3, 0.6] · [0.4, 0.7]
                     = (0.3 × 0.4) + (0.6 × 0.7)
                     = 0.12 + 0.42
                     = 0.54

Score("ate" → "mouse") = Query("ate") · Key("mouse")
                       = [0.3, 0.6] · [0.9, 0.1]
                       = (0.3 × 0.9) + (0.6 × 0.1)
                       = 0.27 + 0.06
                       = 0.33

So our raw scores are:
"Cat":   0.36
"ate":   0.54
"mouse": 0.33
```

## Step 3: Scale the Scores
We divide by √(dimension of key) to prevent scores from getting too large:
```
d_k = 2 (dimension of our key vectors)
√d_k = √2 ≈ 1.414

Scaled scores:
"Cat":   0.36 / 1.414 = 0.255
"ate":   0.54 / 1.414 = 0.382
"mouse": 0.33 / 1.414 = 0.233
```

## Step 4: Apply Softmax (Convert to Percentages)
This is the key step that converts raw scores into percentages that sum to 100%.
The softmax function does this:
```py
softmax(x_i) = e^(x_i) / Σ(e^(x_j))
```
Let's calculate:
```
e^0.255 = 1.29
e^0.382 = 1.47
e^0.233 = 1.26

Sum = 1.29 + 1.47 + 1.26 = 4.02

Final percentages:
"Cat":   1.29 / 4.02 = 0.32 = 32%
"ate":   1.47 / 4.02 = 0.37 = 37%
"mouse": 1.26 / 4.02 = 0.31 = 31%
```
These are our attention weights!

(Note: I simplified the example earlier with 40%, 20%, 40% for clarity, but this is the real calculation)

## Why Dot Product Measures Similarity?
The dot product is high when two vectors point in similar directions:
```
Vector A = [1, 0]
Vector B = [1, 0]
A · B = 1  (very similar!)

Vector A = [1, 0]
Vector C = [0, 1]
A · C = 0  (orthogonal, not similar)

Vector A = [1, 0]
Vector D = [-1, 0]
A · D = -1  (opposite directions!)
```

### The Learning Process
Important: The W_Q, W_K, W_V matrices are learned during training through backpropagation!

During training:

- Model makes predictions
- Compares to correct answers
- Adjusts W_Q, W_K, W_V to improve
- Over time, learns useful patterns like:
  - Verbs should attend to their subjects and objects
  - Pronouns should attend to the nouns they refer to
  - Adjectives should attend to the nouns they modify

## Complete Example with Actual Numbers
Let me show you a full calculation:
```python
import numpy as np

# Word embeddings (simplified - usually 512 or more dimensions)
embeddings = {
    "Cat":   np.array([0.2, 0.8, 0.3]),
    "ate":   np.array([0.5, 0.4, 0.9]),
    "mouse": np.array([0.1, 0.7, 0.6])
}

# Learned weight matrices (3x2 to reduce dimension to 2)
W_Q = np.array([[0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6]])

W_K = np.array([[0.7, 0.8],
                [0.9, 0.1],
                [0.2, 0.3]])

W_V = np.array([[0.4, 0.5],
                [0.6, 0.7],
                [0.8, 0.9]])

# Step 1: Create Q, K, V for each word
def create_qkv(word):
    emb = embeddings[word]
    q = emb @ W_Q
    k = emb @ W_K
    v = emb @ W_V
    return q, k, v

# For the word "ate"
q_ate, k_ate, v_ate = create_qkv("ate")
_, k_cat, v_cat = create_qkv("Cat")
_, k_mouse, v_mouse = create_qkv("mouse")

print("Query for 'ate':", q_ate)
print("Key for 'Cat':", k_cat)
print("Key for 'ate':", k_ate)
print("Key for 'mouse':", k_mouse)

# Step 2: Calculate attention scores (dot products)
score_cat = np.dot(q_ate, k_cat)
score_ate = np.dot(q_ate, k_ate)
score_mouse = np.dot(q_ate, k_mouse)

print("\nRaw scores:")
print(f"'ate' → 'Cat': {score_cat:.3f}")
print(f"'ate' → 'ate': {score_ate:.3f}")
print(f"'ate' → 'mouse': {score_mouse:.3f}")

# Step 3: Scale by sqrt(d_k)
d_k = 2
scaled_scores = np.array([score_cat, score_ate, score_mouse]) / np.sqrt(d_k)

print("\nScaled scores:", scaled_scores)

# Step 4: Apply softmax
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

attention_weights = softmax(scaled_scores)

print("\nAttention weights (percentages):")
print(f"'Cat':   {attention_weights[0]:.1%}")
print(f"'ate':   {attention_weights[1]:.1%}")
print(f"'mouse': {attention_weights[2]:.1%}")

# Step 5: Compute weighted sum of values
new_ate = (attention_weights[0] * v_cat + 
           attention_weights[1] * v_ate + 
           attention_weights[2] * v_mouse)

print("\nNew representation of 'ate':", new_ate)
```

## The Key Insight 💡
The percentages are NOT arbitrary - they come from:

- Learned projections (W_Q, W_K, W_V) that the model learns during training
- Dot product similarity between Query and Keys
- Softmax normalization to make them sum to 100%

**The model learns which words should pay attention to which other words by adjusting these weight matrices during training!**

When the model sees "ate" many times in training data with subjects like "Cat", it learns that queries for verbs should have high dot products with keys for their subjects. That's how it "learns" that "ate" should attend strongly to "Cat"! 🎯
