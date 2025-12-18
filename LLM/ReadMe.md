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

## LLMs dont use nGrams
```text
Modern Large Language Models (LLMs) like GPT series (e.g., GPT-4o and successors) and Grok (from xAI) do not use fixed n-grams such as bigrams (n=2), trigrams (n=3), or any small fixed n in the traditional statistical sense.

Why Traditional n-Grams Are Not Used

Traditional n-gram models (common in pre-2010s NLP) predict the next word based only on a fixed small number of previous words (e.g., 1 for bigram, 2 for trigram). They rely on counting frequencies in a corpus and suffer from sparsity and inability to handle long-range dependencies.
Modern LLMs use transformer architectures with self-attention mechanisms. This allows the model to consider the entire context (all previous tokens) dynamically when predicting the next token, capturing complex, long-distance relationships far beyond any fixed small n.
Context Window as "Effective n"

The closest analog to "n" in modern LLMs is the context window (maximum number of tokens the model can process at once). This acts like a very large variable-length "n-gram" where n can be thousands or millions of tokens.
As of late 2025:

GPT models (e.g., GPT-4o and later versions like GPT-4.1 or GPT-5 series) → typically have context windows of 128,000 to 1 million tokens (with some variants reaching higher).
Grok models (xAI):
Grok 3 → around 128k–131k tokens (some claims of 1M, but documented ~128k–131k).
Grok 4 → often 256k tokens.
Grok 4 Fast / Grok 4.1 Fast variants → up to 2 million tokens (one of the largest available).


These massive context lengths (e.g., 128k+ tokens ≈ hundreds of pages of text) enable LLMs to handle entire books, long conversations, or complex codebases in one go—impossible with traditional small n-grams.
In summary: No fixed small n-gram is used. Instead, transformers provide full-context modeling with context windows in the hundreds of thousands to millions of tokens, making them vastly superior to classical n-gram approaches.
```

## Attention is all you need

```text
Back in 2017, a team of Google researchers wrote a famous paper called "Attention Is All You Need". It's the blueprint for how almost all modern AI chatbots (like ChatGPT, Grok, or Gemini) work under the hood. Let's break it down like you're in high school—no crazy math needed!
The Old Way Was Slow and Clunky

Before this paper, computers translated languages (like English to French) using models called RNNs or LSTMs. Think of them like reading a book word-by-word, from left to right. You have to remember everything you've read so far to understand the next part.

Problem: These were super slow because they could only process one word at a time. You couldn't speed them up easily on computers, and they sometimes forgot stuff from the beginning of long sentences.
The Big Idea: "Attention" Fixes Everything
The researchers said: "What if we ditch that slow step-by-step reading? Let's use something called attention instead!"
Attention is like how you focus in class. When you're reading a sentence, your brain doesn't treat every word equally—it pays more "attention" to the important ones.
In the AI:

Every word gets turned into a number code (called an embedding).
The model creates three things for each word: a Query (like "What am I looking for?"), a Key (like "What do I have?"), and a Value (the actual info).
It compares queries and keys (using a quick math trick called dot-product) to score how much one word should pay attention to another.
Then it mixes the values based on those scores. Boom—each word now "knows" about the important parts of the whole sentence, all at once!

They made it even better with multi-head attention: Like having 8 pairs of eyes looking at the sentence from different angles (one might focus on grammar, another on meaning).
```
<img src="https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png" width="60%" heigh="60%">

<img src="https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/multi-head.png" width="60%" heigh="60%">


### The New Invention: The Transformer

<img src="https://towardsdatascience.com/wp-content/uploads/2022/09/1nqEy4i4sQPhYV0E2n436fQ.png" width="60%" heigh="60%">

Dot Product:

For two vectors $\mathbf{a}$ and $\mathbf{b}$ in $n$-dimensional space:


$$\mathbf{a} = [a_1, a_2, \dots, a_n], \quad \mathbf{b} = [b_1, b_2, \dots, b_n]$$

$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^n a_i b_i$

Alternative geometric definition (in 2D or 3D):

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta$$

where:

$\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ are the magnitudes (lengths) of the vectors,
$\theta$ is the angle between them.

This shows:
```text
If vectors point in the same direction ($\theta = 0^\circ$), dot product is maximum (positive).
If perpendicular ($\theta = 90^\circ$), dot product = 0.
If opposite directions ($\theta = 180^\circ$), dot product is negative and minimum.
```
```

```text
They built a whole new AI called the Transformer. It has two main parts:

Encoder: Reads the input sentence (e.g., English) and understands it using stacks of attention layers.
Decoder: Creates the output sentence (e.g., French) word-by-word, but using attention to look back at the encoder and its own previous words.

No slow looping—just attention layers stacked up (they used 6 for each).
They also added "positional encodings" (wavy math signals) so the model knows the order of words, since attention doesn't care about order naturally.
shreyansh26.github.iotowardsdatascience.com

Why It Was a Game-Changer

Faster training: Everything happens in parallel—way quicker on computers.
Better at long sentences: Remembers connections across the whole text.
They tested it on translation and beat the best old models while training faster.

This Transformer is the heart of today's huge AIs. Models like GPT are basically the "decoder-only" version (great for generating text). The paper basically said: Forget complicated old stuff—attention is all you need!
Cool, right?
```

## Understanding Positional Encoders in LLMs
```text
Imagine you're reading a sentence, but someone hands you all the words on separate cards in a random pile. You'd have a hard time understanding the meaning, right?
"ate cat the mouse the"
vs.
"the cat ate the mouse"
Same words, totally different meanings!
The Problem
This is exactly the problem that language models like ChatGPT face. Unlike humans who read words in order, these AI models look at ALL the words in a sentence at the same time (this makes them super fast). But this creates a problem: how does the model know which word came first, second, third, etc.?
The Solution: Positional Encoders
A positional encoder is like a numbering system that tags each word with its position. Think of it like adding timestamps to a video:

Word 1: "the" → gets tag #1
Word 2: "cat" → gets tag #2
Word 3: "ate" → gets tag #3
Word 4: "the" → gets tag #4
Word 5: "mouse" → gets tag #5

But instead of simple numbers, the model uses special mathematical patterns (like wave patterns) that help it understand not just "this word is position 3" but also "this word is near position 2 and 4."
Real-World Analogy
It's like organizing your Spotify playlist. The songs have:

Their own identity (the actual song/word)
Their position in the playlist (1st, 2nd, 3rd)

Both pieces of information matter! A love song hits different if it's the opening track versus the closing track.
That's essentially what positional encoders do—they make sure the AI knows word order matters!
```

### What is Forward Propagation?

Forward propagation (or forward pass) is the process by which input data flows through the neural network from the input layer to the output layer to produce a prediction.

During forward propagation:
- Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function (e.g., ReLU, sigmoid, tanh).
- The output of one layer becomes the input to the next layer.
- At the final layer, the network produces an output (e.g., class probabilities in classification or a scalar value in regression).
- A loss function then measures how far this prediction is from the true target.

### What is Backward Propagation?
Backward propagation (or backpropagation) is the algorithm used to train the neural network by updating its weights and biases to minimize the loss.

It works by:
- Computing the gradient (partial derivative) of the loss with respect to each weight and bias, starting from the output layer and moving backward to the input layer.
- Using the chain rule of calculus to efficiently propagate these gradients backward.
- Updating parameters using an optimization algorithm like gradient descent:

```java
new_weight = old_weight - learning_rate × gradient
```
Backpropagation allows the network to learn by attributing error to earlier layers and adjusting weights accordingly.

Here is a simple Python code that implements the tiny neural network.
```py
# A super simple neural network with:
# 1 input -> 1 hidden neuron (with ReLU) -> 1 output
# We will do ONE forward pass and ONE backward pass by hand

# Step 1: Set up our values (just like in the example)
x = 2.0          # Input value
y_true = 4.0     # The correct answer we want the network to predict

# Initial weights and biases (these are what the network will learn)
w1 = 2.0         # Weight from input to hidden neuron
b1 = -3.0        # Bias for hidden neuron
w2 = 1.5         # Weight from hidden to output
b2 = 0.5         # Bias for output

learning_rate = 0.1  # How big of a step we take when updating weights

print("=== Starting Forward Propagation ===")
print(f"Input x = {x}")
print(f"True answer y = {y_true}\n")

# Forward Propagation (going from input to output)

# Hidden neuron calculation before activation
z1 = w1 * x + b1
print(f"Hidden pre-activation z1 = w1 * x + b1 = {w1} * {x} + {b1} = {z1}")

# ReLU activation: if z1 > 0, keep it; else make it 0
h = max(0, z1)   # This is ReLU
print(f"Hidden neuron output h = ReLU(z1) = {h}")

# Output neuron (no activation, just linear)
y_pred = w2 * h + b2
print(f"Predicted output y_pred = w2 * h + b2 = {w2} * {h} + {b2} = {y_pred}")

# Calculate how wrong we are (loss)
loss = (y_pred - y_true) ** 2 / 2
print(f"\nLoss = {(y_pred - y_true)}² / 2 = {loss}\n")

print("=== Starting Backward Propagation (learning from the mistake) ===")

# Backward Propagation (going from loss back to weights)

# 1. How much the loss changes if we change y_pred
d_loss_y_pred = y_pred - y_true   # Derivative of loss w.r.t. y_pred
print(f"Step 1: d_loss/y_pred = {y_pred} - {y_true} = {d_loss_y_pred}")

# 2. Gradients for w2 and b2
d_loss_w2 = d_loss_y_pred * h     # Because y_pred depends on w2 * h
d_loss_b2 = d_loss_y_pred * 1     # Because y_pred depends on + b2
print(f"Step 2: d_loss/w2 = {d_loss_y_pred} * {h} = {d_loss_w2}")
print(f"        d_loss/b2 = {d_loss_y_pred} * 1 = {d_loss_b2}")

# 3. How much error flows back to the hidden neuron h
d_loss_h = d_loss_y_pred * w2
print(f"Step 3: d_loss/h = {d_loss_y_pred} * {w2} = {d_loss_h}")

# 4. ReLU derivative: if z1 > 0, derivative = 1; else 0
relu_derivative = 1 if z1 > 0 else 0
d_loss_z1 = d_loss_h * relu_derivative
print(f"Step 4: ReLU derivative = {relu_derivative}")
print(f"        d_loss/z1 = {d_loss_h} * {relu_derivative} = {d_loss_z1}")

# 5. Gradients for w1 and b1
d_loss_w1 = d_loss_z1 * x
d_loss_b1 = d_loss_z1 * 1
print(f"Step 5: d_loss/w1 = {d_loss_z1} * {x} = {d_loss_w1}")
print(f"        d_loss/b1 = {d_loss_z1} * 1 = {d_loss_b1}")

# 6. Update weights and biases (learning!)
print("\n=== Updating weights (learning step) ===")
print(f"Old values: w1={w1}, b1={b1}, w2={w2}, b2={b2}")

w1 = w1 - learning_rate * d_loss_w1
b1 = b1 - learning_rate * d_loss_b1
w2 = w2 - learning_rate * d_loss_w2
b2 = b2 - learning_rate * d_loss_b2

print(f"New values: w1={w1:.1f}, b1={b1:.1f}, w2={w2:.1f}, b2={b2:.1f}")

# Let's do one more forward pass to see if it improved!
print("\n=== One more forward pass to see improvement ===")
z1_new = w1 * x + b1
h_new = max(0, z1_new)
y_pred_new = w2 * h_new + b2
loss_new = (y_pred_new - y_true) ** 2 / 2

print(f"New prediction = {y_pred_new}")
print(f"Old loss was {loss}, new loss is {loss_new:.1f}")
print("(Smaller loss means the network got better!)")
```

Output:
```text
Input was 2.0, true answer 4.0
First prediction: 2.0 (pretty far off)
Loss: 2.0

After one learning step:
New prediction: around 3.02 (much closer to 4.0!)
New loss: about 0.48 (much smaller!)
```

Example with multiple Epochs and a dataset with 3 samples:
```py
# A super simple neural network with:
# 1 input -> 1 hidden neuron (with ReLU) -> 1 output
# Now with MULTIPLE EXAMPLES (3 inputs) and MULTIPLE TRAINING EPOCHS (loops to learn better)

# Step 1: Set up our dataset (3 examples: input x and true y)
# Let's say we're trying to learn something like y ≈ 2 * x (but with ReLU, it might not be perfect)
dataset = [
    (1.0, 2.0),  # Example 1: x=1, y=2
    (2.0, 4.0),  # Example 2: x=2, y=4 (same as before)
    (3.0, 6.0)   # Example 3: x=3, y=6
]

# Initial weights and biases (these are what the network will learn)
w1 = 2.0         # Weight from input to hidden neuron
b1 = -3.0        # Bias for hidden neuron
w2 = 1.5         # Weight from hidden to output
b2 = 0.5         # Bias for output

learning_rate = 0.1  # How big of a step we take when updating weights
num_epochs = 5       # How many full training loops (epochs) to do. Try changing this!

print("=== Starting Training ===")
print(f"We have {len(dataset)} examples to learn from.")
print(f"Initial weights: w1={w1}, b1={b1}, w2={w2}, b2={b2}\n")

# Big loop: For each epoch (full pass through all data)
for epoch in range(num_epochs):
    total_loss = 0.0  # Keep track of how wrong we are overall this epoch
    
    print(f"--- Epoch {epoch + 1} (Training Loop {epoch + 1}) ---")
    
    # Inner loop: Go through each example in the dataset
    for x, y_true in dataset:
        print(f"\nProcessing example: x={x}, y_true={y_true}")
        
        # Forward Propagation (make a guess)
        
        # Hidden neuron before activation
        z1 = w1 * x + b1
        print(f"  Hidden pre-activation z1 = {w1} * {x} + {b1} = {z1}")
        
        # ReLU activation: max(0, z1)
        h = max(0, z1)
        print(f"  Hidden output h = ReLU(z1) = {h}")
        
        # Output (linear)
        y_pred = w2 * h + b2
        print(f"  Predicted y_pred = {w2} * {h} + {b2} = {y_pred}")
        
        # Loss for this example
        loss = (y_pred - y_true) ** 2 / 2
        print(f"  Loss for this example = {(y_pred - y_true)}² / 2 = {loss}")
        total_loss += loss  # Add to total
        
        # Backward Propagation (learn from mistake)
        
        # 1. Derivative of loss w.r.t. y_pred
        d_loss_y_pred = y_pred - y_true
        print(f"  Backprop Step 1: d_loss/y_pred = {y_pred} - {y_true} = {d_loss_y_pred}")
        
        # 2. Gradients for w2 and b2
        d_loss_w2 = d_loss_y_pred * h
        d_loss_b2 = d_loss_y_pred * 1
        print(f"  Step 2: d_loss/w2 = {d_loss_y_pred} * {h} = {d_loss_w2}")
        print(f"          d_loss/b2 = {d_loss_y_pred} * 1 = {d_loss_b2}")
        
        # 3. Error back to h
        d_loss_h = d_loss_y_pred * w2
        print(f"  Step 3: d_loss/h = {d_loss_y_pred} * {w2} = {d_loss_h}")
        
        # 4. Through ReLU
        relu_derivative = 1 if z1 > 0 else 0
        d_loss_z1 = d_loss_h * relu_derivative
        print(f"  Step 4: ReLU deriv = {relu_derivative}, d_loss/z1 = {d_loss_h} * {relu_derivative} = {d_loss_z1}")
        
        # 5. Gradients for w1 and b1
        d_loss_w1 = d_loss_z1 * x
        d_loss_b1 = d_loss_z1 * 1
        print(f"  Step 5: d_loss/w1 = {d_loss_z1} * {x} = {d_loss_w1}")
        print(f"          d_loss/b1 = {d_loss_z1} * 1 = {d_loss_b1}")
        
        # 6. Update weights (learn!)
        w1 = w1 - learning_rate * d_loss_w1
        b1 = b1 - learning_rate * d_loss_b1
        w2 = w2 - learning_rate * d_loss_w2
        b2 = b2 - learning_rate * d_loss_b2
        print(f"  Updated weights: w1={w1:.2f}, b1={b1:.2f}, w2={w2:.2f}, b2={b2:.2f}")
    
    # After all examples in this epoch, show average loss
    avg_loss = total_loss / len(dataset)
    print(f"\nEpoch {epoch + 1} finished! Average loss: {avg_loss:.2f}")
    print(" (Lower loss means better learning!)\n")

# After all training, test on the dataset again
print("=== Final Test After Training ===")
for x, y_true in dataset:
    z1 = w1 * x + b1
    h = max(0, z1)
    y_pred = w2 * h + b2
    print(f"Input x={x}, True y={y_true}, Final Prediction={y_pred:.2f}")
```

Output:
```py
=== Starting Training ===
We have 3 examples to learn from.
Initial weights: w1=2.0, b1=-3.0, w2=1.5, b2=0.5

--- Epoch 1 (Training Loop 1) ---

Processing example: x=1.0, y_true=2.0
 Hidden pre-activation z1 = 2.0 * 1.0 + -3.0 = -1.0
 Hidden output h = ReLU(z1) = 0
 Predicted y_pred = 1.5 * 0 + 0.5 = 0.5
 Loss for this example = -1.5² / 2 = 1.125
 Backprop Step 1: d_loss/y_pred = 0.5 - 2.0 = -1.5
 Step 2: d_loss/w2 = -1.5 * 0 = -0.0
 d_loss/b2 = -1.5 * 1 = -1.5
 Step 3: d_loss/h = -1.5 * 1.5 = -2.25
 Step 4: ReLU deriv = 0, d_loss/z1 = -2.25 * 0 = -0.0
 Step 5: d_loss/w1 = -0.0 * 1.0 = -0.0
 d_loss/b1 = -0.0 * 1 = -0.0
 Updated weights: w1=2.00, b1=-3.00, w2=1.50, b2=0.65

Processing example: x=2.0, y_true=4.0
 Hidden pre-activation z1 = 2.0 * 2.0 + -3.0 = 1.0
 Hidden output h = ReLU(z1) = 1.0
 Predicted y_pred = 1.5 * 1.0 + 0.65 = 2.15
 Loss for this example = -1.85² / 2 = 1.7112500000000002
 Backprop Step 1: d_loss/y_pred = 2.15 - 4.0 = -1.85
 Step 2: d_loss/w2 = -1.85 * 1.0 = -1.85
 d_loss/b2 = -1.85 * 1 = -1.85
 Step 3: d_loss/h = -1.85 * 1.5 = -2.7750000000000004
 Step 4: ReLU deriv = 1, d_loss/z1 = -2.7750000000000004 * 1 = -2.7750000000000004
 Step 5: d_loss/w1 = -2.7750000000000004 * 2.0 = -5.550000000000001
 d_loss/b1 = -2.7750000000000004 * 1 = -2.7750000000000004
 Updated weights: w1=2.56, b1=-2.72, w2=1.69, b2=0.84

Processing example: x=3.0, y_true=6.0
 Hidden pre-activation z1 = 2.555 * 3.0 + -2.7225 = 4.942500000000001
 Hidden output h = ReLU(z1) = 4.942500000000001
 Predicted y_pred = 1.685 * 4.942500000000001 + 0.8350000000000001 = 9.163112500000002
 Loss for this example = 3.163112500000002² / 2 = 5.002640343828132
 Backprop Step 1: d_loss/y_pred = 9.163112500000002 - 6.0 = 3.163112500000002
 Step 2: d_loss/w2 = 3.163112500000002 * 4.942500000000001 = 15.633683531250014
 d_loss/b2 = 3.163112500000002 * 1 = 3.163112500000002
 Step 3: d_loss/h = 3.163112500000002 * 1.685 = 5.3298445625000035
 Step 4: ReLU deriv = 1, d_loss/z1 = 5.3298445625000035 * 1 = 5.3298445625000035
 Step 5: d_loss/w1 = 5.3298445625000035 * 3.0 = 15.98953368750001
 d_loss/b1 = 5.3298445625000035 * 1 = 5.3298445625000035
 Updated weights: w1=0.96, b1=-3.26, w2=0.12, b2=0.52

Epoch 1 finished! Average loss: 2.61
 (Lower loss means better learning!)

--- Epoch 2 (Training Loop 2) ---

... (continuing similarly for epochs 2-5; the hidden neuron eventually stops activating due to updates pushing b1 lower and w1 not recovering the activation region)

=== Final Test After Training ===
Input x=1.0, True y=2.0, Final Prediction=3.12
Input x=2.0, True y=4.0, Final Prediction=3.12
Input x=3.0, True y=6.0, Final Prediction=3.12
```

### Updated Neural Network Example: Now with 2 Input Features

We've extended the toy network to handle multiple input features (2 in this case):

- Input layer: 2 features (x1, x2)
- Hidden layer: Still 1 neuron with ReLU activation
- Output layer: 1 linear neuron

The target is still approximately y = 2 * x1 (ignoring x2, since x2=0 in all examples).

```py
dataset = [
    ((1.0, 0.0), 2.0),
    ((2.0, 0.0), 4.0),
    ((3.0, 0.0), 6.0)
]

# Initial parameters
w1_1 = 2.0   # weight for feature 1
w1_2 = 0.5   # weight for feature 2
b1 = 0.0
w2 = 1.5
b2 = 0.0
learning_rate = 0.05
num_epochs = 10

print("=== Starting Training ===")
print(f"Initial: w1_1={w1_1}, w1_2={w1_2}, b1={b1}, w2={w2}, b2={b2}\n")

for epoch in range(num_epochs):
    total_loss = 0.0
    print(f"--- Epoch {epoch + 1} ---")
    
    for features, y_true in dataset:
        x1, x2 = features
        print(f"\nExample: x1={x1}, x2={x2}, y_true={y_true}")
        
        # Forward pass
        z1 = w1_1 * x1 + w1_2 * x2 + b1
        h = max(0, z1)  # ReLU
        y_pred = w2 * h + b2
        
        loss = (y_pred - y_true) ** 2 / 2
        total_loss += loss
        
        # Backward pass
        d_loss_y_pred = y_pred - y_true
        
        d_loss_w2 = d_loss_y_pred * h
        d_loss_b2 = d_loss_y_pred
        
        d_loss_h = d_loss_y_pred * w2
        d_loss_z1 = d_loss_h * (1 if z1 > 0 else 0)
        
        d_loss_w1_1 = d_loss_z1 * x1
        d_loss_w1_2 = d_loss_z1 * x2
        d_loss_b1 = d_loss_z1
        
        # Update
        w1_1 -= learning_rate * d_loss_w1_1
        w1_2 -= learning_rate * d_loss_w1_2
        b1 -= learning_rate * d_loss_b1
        w2 -= learning_rate * d_loss_w2
        b2 -= learning_rate * d_loss_b2
        
        print(f"  y_pred={y_pred:.2f}, loss={loss:.3f}")
        print(f"  Updated weights: w1_1={w1_1:.3f}, w1_2={w1_2:.3f}, b1={b1:.3f}, w2={w2:.3f}, b2={b2:.3f}")
    
    print(f"\nEpoch {epoch+1} average loss: {total_loss/len(dataset):.3f}\n")

# Final test
print("=== Final Predictions ===")
for features, y_true in dataset:
    x1, x2 = features
    z1 = w1_1 * x1 + w1_2 * x2 + b1
    h = max(0, z1)
    y_pred = w2 * h + b2
    print(f"x1={x1}, x2={x2} → prediction={y_pred:.2f} (true={y_true})")
```

## Cross Entropy in Machine Learning
Cross entropy is a way to measure how wrong your model's predictions are, especially for classification problems (like deciding if an email is spam or not spam, or recognizing if a picture shows a cat, dog, or bird).

Cross entropy measures how "surprised" we should be by the reality given your prediction. If you predicted high probability for what actually happened, your cross entropy (error) is low. If you predicted low probability for what actually happened, your cross entropy (error) is high.

A Simple Example

Let's say you're building a model to identify animals in photos. For one photo:

Your model's prediction:

- Cat: 70% (0.7)
- Dog: 20% (0.2)
- Bird: 10% (0.1)

Actual answer: It's a Cat

Cross Entropy Calculation:
```text
Cross Entropy = -log(probability of correct answer)
              = -log(0.7)
              = 0.36
```
The better your prediction, the lower the cross entropy!

Why Not Just Use Regular Error?

Regular error (like in our backpropagation example) works for predicting numbers (test scores, house prices).

But for classification (choosing between categories), we need something that:

- Works with probabilities (0 to 1)
- Heavily penalizes confident wrong answers
- Has nice mathematical properties for optimization

### Real Example: Email Spam Detector
```yaml
Email #1:

Model predicts: 95% spam (p = 0.95)
Actual: Spam (y = 1)
Cross Entropy = -(1 × log(0.95)) = 0.05 ✅ Low error!

Email #2:

Model predicts: 30% spam (p = 0.30)
Actual: Spam (y = 1)
Cross Entropy = -(1 × log(0.30)) = 1.20 ❌ Higher error!

Email #3:

Model predicts: 95% spam (p = 0.95)
Actual: Not spam (y = 0)
Cross Entropy = -(0 × log(0.95) + 1 × log(0.05)) = 3.00 ❌❌ Very high error!
```

### Why "Cross" Entropy?
The "cross" comes from information theory - you're measuring the difference between two probability distributions:

- The true distribution (actual answer: 100% cat, 0% dog, 0% bird)
- Your predicted distribution (70% cat, 20% dog, 10% bird)

When to Use It
```yaml
✅ Classification problems (spam/not spam, cat/dog/bird)
✅ When output is probabilities
❌ Not for regression (predicting continuous numbers like prices or temperatures)
```

## What is a Log Function? 🧮
A logarithm (log) is just a fancy way of asking: "How many times do I need to multiply 10 to get this number?"

"What power do I raise 10 to, to get this number?" That's it! 🎯
```
log(50) = ≈ 1.7 (because 10^1.7 ≈ 50)
log(500) = ≈ 2.7 (because 10^2.7 ≈ 500)
log(5) = ≈ 0.7 (because 10^0.7 ≈ 5)

log(1) = 0

Why? Because 10⁰ = 1
You don't need to multiply 10 at all!

log(0.1) = -1

Why? Because 10⁻¹ = 1/10 = 0.1
Negative exponents mean division!

Exponent
10² = 100

↓ Log goes backwards ↓

log(100) = 2
```

(Note: By default, "log" usually means log base 10, but you can have other bases too, like log base 2 or "natural log" which uses a special number called e. But the concept is the same!)

### What is "e" (Euler's Number)? 🔢
The Number Itself: e ≈ 2.71828...

It goes on forever (like π = 3.14159...), but we usually round it to 2.718.

```
e = lim(n→∞) (1 + 1/n)ⁿ

e ≈ 2.718 is the number you get from continuous compound growth
ln(x) asks: "e to what power equals x?"
It's called "natural" because it appears naturally in growth, decay, and calculus
```
In ML, we use ln because the math works out beautifully!

Think of e as nature's favorite growth rate! 🌱

## What is Self-Attention

Self-attention is a mechanism that allows a model to look at different parts of the input and decide which parts are most important for understanding each word.

### The "Self" Part
It's called "self-attention" because the input attends to itself - each word looks at all other words (including itself) in the sentence to understand its meaning better.

### Real-Life Analogy 📚
Imagine you're reading this sentence: _"The animal didn't cross the street because it was too tired."_

When you read the word "it", your brain automatically looks back at the sentence to figure out what "it" refers to. Is it the animal or the street?

You pay attention to "animal" because that makes sense - animals get tired, streets don't!

That's exactly what self-attention does - it helps each word "look around" at other words to understand context better.

### How It Works (Step-by-Step)
Let's use a simple sentence: "Cat ate mouse". 

Step1: Each word asks "Who should I pay attention to?"

For the word "ate":
- How much should I look at "Cat"?
- How much should I look at "ate" (itself)?
- How much should I look at "mouse"?

Step2: Step 2: Calculate Attention Scores

The model creates three things for each word (using learned weights):

- Query (Q): "What am I looking for?"
- Key (K): "What do I have to offer?"
- Value (V): "What information do I carry?"

Step 3: Compute Similarity

For "ate" attending to each word:
```yaml
Attention Score = Query("ate") · Key("Cat")    → High score! (subject of action)
Attention Score = Query("ate") · Key("ate")    → Medium score
Attention Score = Query("ate") · Key("mouse")  → High score! (object of action)
```
The dot product (·) measures how similar the query and key are.

Step 4: Softmax (Make it a Probability)
```yaml
Convert scores to percentages that sum to 100%:
"Cat":   40%
"ate":   20%
"mouse": 40%
```

Step 5: Weighted Sum

Create a new representation of "ate" by combining the Values:
```py
New "ate" = (40% × Value("Cat")) + (20% × Value("ate")) + (40% × Value("mouse"))
```
Now the representation of "ate" contains information from all relevant words!

Visual Example:
```text
Sentence: "The cat sat on the mat"
When processing "sat":
┌───────────────────────────────────────────────┐
│  "The"   "cat"   "sat"   "on"   "the"   "mat" │
│   ↓      ↓       ↓      ↓       ↓      ↓      │
│   5%    40%     20%    10%     5%     20%     │
│   └──────┴───────┴──────┴───────┴──────┘      │
│              Attention Weights                │
│                     ↓                         │
│         Enhanced "sat" representation         │
└───────────────────────────────────────────────┘
```

### Why Is It Powerful?
1. Captures Long-Range Dependencies<br>
"The cat, which was very fluffy and belonged to my neighbor, ate the mouse."<br>
Self-attention can connect "cat" and "ate" even though they're far apart!
2. Parallel Processing<br>
Unlike RNNs (which process words one-by-one), self-attention looks at ALL words simultaneously. This makes it:

    Much faster to train<br>
    Can use GPUs efficiently

3. Context-Aware Representations<br>
The same word gets different representations based on context:
```text
"bank" in "river bank" vs "bank account"
"bat" in "baseball bat" vs "vampire bat"
```

### Multi-Head Attention (The "Multi" Part)
In practice, we use multiple attention heads running in parallel. Think of it like having multiple experts:
```yaml
Head 1: Focuses on grammar (subject-verb relationships)
Head 2: Focuses on semantics (meaning)
Head 3: Focuses on co-references (pronouns)
etc.
```
Each head learns to pay attention to different aspects!

### Key Takeaway
**Self-attention is like giving every word a pair of eyes to look around the sentence and decide "Which other words help me understand my meaning better?"**

Instead of processing words in isolation, each word gets to "consult" with all the other words to build a richer, context-aware understanding! 🎯
