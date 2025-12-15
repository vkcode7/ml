## What is a Bigram Language Model?
```text
A bigram language model is a simple statistical model in natural language processing (NLP) that predicts the probability of a word in a sequence based only on the immediately preceding word. It is a type of n-gram model where n=2 (bi- means two).
Key Idea

Language models assign probabilities to sequences of words, e.g., estimating how likely a sentence is.
In a bigram model, the probability of a sentence like "The cat sat" is approximated as:textP(The cat sat) ≈ P(The | <start>) × P(cat | The) × P(sat | cat) × P(<end> | sat)Here, each word's probability depends solely on the previous one (Markov assumption).
Formally, the conditional probability is:textP(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})This is estimated via maximum likelihood estimation from a training corpus (counting how often word pairs appear).

How It Works

Training: Count all pairs of consecutive words (bigrams) in a large text corpus.

Prediction/Generation: To generate text, start with a word and repeatedly sample the next word based on bigram probabilities.
Handling Issues:

Data sparsity: Unseen bigrams get probability 0 → Use smoothing (e.g., add-one/Laplace: add 1 to all counts).
Unknown words: Replace with <UNK> token.

Example
Corpus: "I like to eat. I like pizza."
Bigrams: (I, like), (like, to), (to, eat), etc.

P(like | I) = high (appears twice).
Generated text might be repetitive or nonsensical but follows local patterns (e.g., "I like to like...").

Advantages

Simple, fast, and easy to implement.
Captures basic word co-occurrences (better than unigram models, which treat words independently).

Limitations

Only short context (one previous word) → Ignores long-range dependencies.
Struggles with larger vocabularies (sparse data).
Modern alternatives like trigrams (n=3) or neural models perform better.

Relation to Modern LLMs
Note: The acronym LLM today almost always means Large Language Model (e.g., GPT series), which are massive neural networks (transformers) trained on billions of words, capturing long contexts and complex patterns.
A bigram model is not an LLM—it's a basic statistical precursor from early NLP. Modern LLMs build on ideas like n-grams but use deep learning for vastly superior performance. Bigram models are mainly educational or used in simple applications like autocomplete or speech recognition tuning.
Bigram models laid the foundation for understanding probabilistic language modeling, which evolved into today's powerful LLMs.
```

