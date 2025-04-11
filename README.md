# ml
### ML projects

#### How to create a virtual env in python?
To use venv in your project, in your terminal, create a new project folder, cd to the project folder in your terminal, and run the following command:
```bash
python<version> -m venv virtual-environment-name
  
mkdir projectA
  
cd projectA
  
python3.8 -m venv env
 ```

When you check the new projectA folder, you will notice that a new folder called env has been created. env is the name of our virtual environment, but it can be named anything you want.
<br>
Activate it using:
```bash
source env/bin/activate OR conda activate myvenv
  
deactivate:
deactivate
```
important commands:
```bash
  pip freeze > requirements.txt
  ~ pip install -r requirements.txt
```
To use this virtual env in Spyder: Preferences -> Python Interpreter<br>
https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment
  
Cloning a GIT reopo: 
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```
James Briggs: https://github.com/jamescalam/transformers/tree/main
<br>
Learn next-generation NLP with transformers for sentiment analysis, Q&A, similarity search, NER, and more

### LangChain + Retrieval Local LLMs for Retrieval QA - No OpenAI!!!
this video go through how to use LangChain without Open AI for retrieval QA - Flan-T5, FastChat-T5, StableVicuna, WizardLM
https://www.youtube.com/watch?v=9ISVjh8mdlA


# Chip Huyen: AI Engineering
## Ch 1: Introduction

### Adapting LLMs

This list highlights various techniques that you can use to tailor the LLM to meet your specific needs:
- Prompt engineering is the process of optimizing text prompts to guide the LLM to generate pertinent responses.
- In-context learning (ICL) allows the model to dynamically update its understanding during a conversation, resulting in more contextually relevant responses. Data is fed within the prompt.
- Retrieval-augmented generation (RAG) combines retrieval and generation models to surface new relevant data as part of a prompt.
- Fine-tuning entails customizing a pretrained LLM to enhance its performance for specific domains or tasks.
- Reinforcement learning from human feedback (RLHF) is the ongoing approach to fine-tuning in near real time by providing the model with feedback from human evaluators who guide and improve its responses.

### Token
The basic unit of a language model is token. A token can be a character, a word, or a
part of a word (like -tion), depending on the model.2 For example, GPT-4, a model
behind ChatGPT, breaks the phrase “I can’t wait to build AI applications” into nine
tokens. Note that in this example, the word “can’t” is broken
into two tokens, can and ’t.

The process of breaking the original text into tokens is called tokenization. For
GPT-4, an average token is approximately ¾ the length of a word. So, 100 tokens are
approximately 75 words.
The set of all tokens a model can work with is the model’s vocabulary.

The Mixtral 8x7B
model has a vocabulary size of 32,000. GPT-4’s vocabulary size is 100,256. The tokenization
method and vocabulary size are decided by model developers.

Why do language models use token as their unit instead of word or
character?<br>

- 1. Compared to characters, tokens allow the model to break
words into meaningful components. For example, “cooking”
can be broken into “cook” and “ing”, with both components
carrying some meaning of the original word.
- 2. Because there are fewer unique tokens than unique words, this
reduces the model’s vocabulary size, making the model more
efficient (as discussed in Chapter 2).
- 3. Tokens also help the model process unknown words. For
instance, a made-up word like “chatgpting” could be split into
“chatgpt” and “ing”, helping the model understand its structure.
Tokens balance having fewer units than words while
retaining more meaning than individual characters.

### Masked vs Autoregressive
There are two main types of language models: masked language models and autoregressive
language models. They differ based on what information they can use to predict
a token:

#### Masked language model
A masked language model is trained to predict missing tokens anywhere in a
sequence, using the context from both before and after the missing tokens. In
essence, a masked language model is trained to be able to fill in the blank. For
example, given the context, “My favorite __ is blue”, a masked language model
should predict that the blank is likely “color”. A well-known example of a
masked language model is bidirectional encoder representations from transformers,
or BERT (Devlin et al., 2018).

As of writing, masked language models are commonly used for non-generative
tasks such as sentiment analysis and text classification. 

#### Autoregressive language model
An autoregressive language model is trained to predict the next token in a
sequence, using only the preceding tokens. It predicts what comes next in “My
favorite color is __.” An autoregressive model can continually generate one
token after another. Today, autoregressive language models are the models of
choice for text generation, and for this reason, they are much more popular than
masked language models.

The outputs of language models are open-ended. A language model can use its fixed,
finite vocabulary to construct infinite possible outputs. A model that can generate
open-ended outputs is called generative, hence the term generative AI.
You can think of a language model as a completion machine: given a text (prompt), it
tries to complete that text. Here’s an example:
Prompt (from user): “To be or not to be”
Completion (from language model): “, that is the question.”
It’s important to note that completions are predictions, based on probabilities, and
not guaranteed to be correct. This probabilistic nature of language models makes
them both so exciting and frustrating to use.

### Self-Supervision
Language modeling is just one of many ML algorithms. There are also models for
object detection, topic modeling, recommender systems, weather forecasting, stock
price prediction, etc. What’s special about language models that made them the center
of the scaling approach that caused the ChatGPT moment?
The answer is that language models can be trained using self-supervision, while many
other models require supervision. Supervision refers to the process of training ML
algorithms using labeled data.

Self-supervision helps overcome the data labeling bottleneck. In self-supervision,
instead of requiring explicit labels, the model can infer labels from the input data.
Language modeling is self-supervised because each input sequence provides both the
labels (tokens to be predicted) and the contexts the model can use to predict these
labels.

Self-supervised learning means that language models can learn from text sequences
without requiring any labeling. Because text sequences are everywhere—in books,
blog posts, articles, and Reddit comments—it’s possible to construct a massive
amount of training data, allowing language models to scale up to become LLMs.

A model’s size is typically measured by its number of parameters. A parameter is a variable
within an ML model that is updated through the training process.7 In general,
though this is not always true, the more parameters a model has, the greater its
capacity to learn desired behaviors.

### From Large Language Models to Foundation Models
While many people still call Gemini and GPT-4V LLMs, they’re better characterized
as foundation models. The word foundation signifies both the importance of these
models in AI applications and the fact that they can be built upon for different needs.

A model that can work with more than one data modality is also called a multimodal
model. A generative multimodal model is also called a large multimodal model
(LMM).

### Prompt Engineering / RAG/ Finetune
Using a database to supplement the instructions is called retrievalaugmented
generation (RAG). You can also finetune—further train—the model on a
dataset of high-quality product descriptions.
Prompt engineering, RAG, and finetuning are three very common AI engineering
techniques that you can use to adapt a model to your needs.


### On the Differences Among Training, Pre-Training, Finetuning, and Post-Training

Training always involves changing model weights, but not all changes to model
weights constitute training. For example, quantization, the process of reducing the
precision of model weights, technically changes the model’s weight values but isn’t
considered training.

The term training can often be used in place of pre-training, finetuning, and posttraining,
which refer to different training phases:

#### Pre-training
Pre-training refers to training a model from scratch—the model weights are randomly
initialized. For LLMs, pre-training often involves training a model for text
completion. Out of all training steps, pre-training is often the most resourceintensive
by a long shot. For the InstructGPT model, pre-training takes up to
98% of the overall compute and data resources. Pre-training also takes a long
time to do. A small mistake during pre-training can incur a significant financial
loss and set back the project significantly. Due to the resource-intensive nature of
pre-training, this has become an art that only a few practice. Those with expertise
in pre-training large models, however, are heavily sought after.

#### Finetuning
Finetuning means continuing to train a previously trained model—the model
weights are obtained from the previous training process. Because the model
already has certain knowledge from pre-training, finetuning typically requires
fewer resources (e.g., data and compute) than pre-training.

#### Post-training
Many people use post-training to refer to the process of training a model after the
pre-training phase. Conceptually, post-training and finetuning are the same and
can be used interchangeably. However, sometimes, people might use them differently
to signify the different goals. It’s usually post-training when it’s done by
model developers. For example, OpenAI might post-train a model to make it
better at following instructions before releasing it. It’s finetuning when it’s done
by application developers. For example, you might finetune an OpenAI model
(which might have been post-trained itself) to adapt it to your needs.

### Inference optimization. 
Inference optimization means making models faster and
cheaper.

### Prompt engineering and context construction.
Prompt engineering is about getting AI
models to express the desirable behaviors from the input alone, without changing the
model weights.

It’s possible to get a model to do amazing things with just prompts. The right instructions
can get a model to perform the task you want, in the format of your choice.
Prompt engineering is not just about telling a model what to do. It’s also about giving
the model the necessary context and tools to do a given task. For complex tasks with
long context, you might also need to provide the model with a memory management
system so that the model can keep track of its history.

### Language Support
Before foundation models, the most
popular ML frameworks supported mostly Python APIs. Today, Python is still popular but there is also increasing support for JavaScript APIs, with LangChain.js,
Transformers.js, OpenAI’s Node library, and Vercel’s AI SDK.

## Ch 2: Understanding Fpundation Models

### Training Data
An AI model is only as good as the data it was trained on.

A common source for training data is Common Crawl, created by a
nonprofit organization that sporadically crawls websites on the internet. In 2022 and
2023, this organization crawled approximately 2–3 billion web pages each month.
Google provides a clean subset

The data quality of Common Crawl, and C4 to a certain extent, is questionable—
think clickbait, misinformation, propaganda, conspiracy theories, racism, misogyny,
and every sketchy website you’ve ever seen or avoided on the internet.

### Multilingual Models
- English accounts for almost half of the data (45.88%), making it eight times more
prevalent than the second-most common language, Russian (5.97%)
- Given the dominance of English in the internet data, it’s not surprising that generalpurpose
models work much better for English than other languages
- Similarly, when tested on six math problems on Project Euler, Yennie Jun found that
GPT-4 was able to solve problems in English more than three times as often compared
to Armenian or Farsi.
- Given that LLMs are generally good at translation, can we just translate all queries
from other languages into English, obtain the responses, and translate them back into
the original language? Many people indeed follow this approach, but it’s not ideal.
First, this requires a model that can sufficiently understand under-represented languages
to translate. Second, translation can cause information loss. For example,
some languages, like Vietnamese, have pronouns to denote the relationship between
the two speakers. When translating into English, all these pronouns are translated
into I and you, causing the loss of the relationship information.
- Other than quality issues, models can also be slower and more expensive for non-
English languages.
- To convey the same meaning, languages like Burmese and Hindi require a lot
more tokens than English or Spanish. For the MASSIVE dataset, the median token
length in English is 7, but the median length in Hindi is 32, and in Burmese, it’s a
whopping 72, which is ten times longer than in English.

### Model Architecture
As of this writing, the most dominant architecture for language-based foundation
models is the transformer architecture (Vaswani et al., 2017), which is based on the
attention mechanism.

#### Transformer architecture
The transformer architecture was popularized on the heels of the success of the seq2seq
(sequence-to-sequence) architecture.

The transformer architecture dispenses with RNNs entirely. With transformers, the
input tokens can be processed in parallel, significantly speeding up input processing.
While the transformer removes the sequential input bottleneck, transformer-based
autoregressive language models still have the sequential output bottleneck.

#### Attention mechanism.
At the heart of the transformer architecture is the attention
mechanism. Understanding this mechanism is necessary to understand how transformer
models work. Under the hood, the attention mechanism leverages key, value,
and query vectors:
- The query vector (Q) represents the current state of the decoder at each decoding
step. Using the same book summary example, this query vector can be thought of
as the person looking for information to create a summary.
- Each key vector (K) represents a previous token. If each previous token is a page
in the book, each key vector is like the page number. Note that at a given decoding
step, previous tokens include both input tokens and previously generated
tokens.
- Each value vector (V) represents the actual value of a previous token, as learned
by the model. Each value vector is like the page’s content.

#### Other model architectures
While the transformer model dominates the landscape, it’s not the only architecture.
Since AlexNet revived the interest in deep learning in 2012, many architectures have
gone in and out of fashion. Seq2seq was in the limelight for four years (2014–2018).
GANs (generative adversarial networks) captured the collective imagination a bit
longer (2014–2019). Compared to architectures that came before it, the transformer
is sticky. It’s been around since 2017.10 How long until something better comes
along?

### Model Size
The number of parameters is usually appended at the end of a model name. For
example, Llama-13B refers to the version of Llama, a model family developed by
Meta, with 13 billion parameters.

In general, increasing a model’s parameters increases its capacity to learn, resulting in
better models. Given two models of the same model family, the one with 13 billion
parameters is likely to perform much better than the one with 7 billion parameters.

The number of parameters helps us estimate the compute resources needed to train
and run this model. For example, if a model has 7 billion parameters, and each
parameter is stored using 2 bytes (16 bits), then we can calculate that the GPU memory
needed to do inference using this model will be at least 14 billion bytes (14 GB).

The number of parameters can be misleading if the model is sparse. A sparse model
has a large percentage of zero-value parameters. A 7B-parameter model that is 90%
sparse only has 700 million non-zero parameters. Sparsity allows for more efficient
data storage and computation. This means that a large sparse model can require less
compute than a small dense model.

A type of sparse model that has gained popularity in recent years is mixture-ofexperts
(MoE) (Shazeer et al., 2017). An MoE model is divided into different groups
of parameters, and each group is an expert. Only a subset of the experts is active for
(used to) process each token.

For example, Mixtral 8x7B is a mixture of eight experts, each expert with seven billion
parameters. If no two experts share any parameter, it should have 8 × 7 billion =
56 billion parameters. However, due to some parameters being shared, it has only
46.7 billion parameters.

The number of tokens in a model’s dataset isn’t the same as its number of training
tokens. The number of training tokens measures the tokens that the model is trained
on. If a dataset contains 1 trillion tokens and a model is trained on that dataset for
two epochs—an epoch is a pass through the dataset—the number of training tokens is
2 trillion.

A more standardized unit for a model’s compute requirement is FLOP, or floating
point operation. FLOP measures the number of floating point operations performed
for a certain task. Google’s largest PaLM-2 model, for example, was trained using 10^22
FLOPs. GPT-3-175B was trained using 3.14 × 10^23 FLOPs,

An NVIDIA H100
NVL GPU can deliver a maximum of 60 TeraFLOP/s: 6 × 10^13 FLOPs a second or
5.2 × 10^18 FLOPs a day.

Assume that you have 256 H100s. If you can use them at their maximum capacity
and make no training mistakes, it’d take you (3.14 × 10^23) / (256 × 5.2 × 10^18)
= ~236 days, or approximately 7.8 months, to train GPT-3-175B. 

At 70% utilization and $2/h for one H100, training GPT-3-175B would cost over $4
million:
$2/H100/hour × 256 H100 × 24 hours × 256 days / 0.7 = $4,142,811.43

In summary, three numbers signal a model’s scale:
- Number of parameters, which is a proxy for the model’s learning
capacity.
- Number of tokens a model was trained on, which is a proxy
for how much a model learned.
- Number of FLOPs, which is a proxy for the training cost.

### Parameter Versus Hyperparameter
A parameter can be learned by the model during the training process. A hyperparameter
is set by users to configure the model and control how the model learns.
Hyperparameters to configure the model include the number of layers, the model
dimension, and vocabulary size. Hyperparameters to control how a model learns
include batch size, number of epochs, learning rate, per-layer initial variance, and
more.

### Post-Training
Post-training starts with a pre-trained model. Let’s say that you’ve pre-trained a
foundation model using self-supervision.

Every model’s post-training is different. However, in general, post-training consists
of two steps:

1. Supervised finetuning (SFT): Finetune the pre-trained model on high-quality
instruction data to optimize models for conversations instead of completion.

3. Preference finetuning: Further finetune the model to output responses that align
with human preference.

For language-based foundation models, pre-training optimizes token-level quality,
where the model is trained to predict the next token accurately. However, users don’t
care about token-level quality—they care about the quality of the entire response.
Post-training, in general, optimizes the model to generate responses that users prefer.
Some people compare pre-training to reading to acquire knowledge, while posttraining
is like learning how to use that knowledge.

As post-training consumes a small portion of resources compared to pre-training
(InstructGPT used only 2% of compute for post-training and 98% for pre-training),
you can think of post-training as unlocking the capabilities that the pre-trained
model already has but are hard for users to access via prompting alone.

1. Self-supervised pre-training results in a rogue model that can be considered an
untamed monster because it uses indiscriminate data from the internet.<br>
2. This monster is then supervised finetuned on higher-quality data—Stack Overflow,
Quora, or human annotations—which makes it more socially acceptable.<br>
3. This finetuned model is further polished using preference finetuning to make it
customer-appropriate, which is like giving it a smiley face.<br>

### Preference Finetuning
With great power comes great responsibilities. A model that can assist users in
achieving great things can also assist users in achieving terrible things. Demonstration
data teaches the model to have a conversation but doesn’t teach the model what
kind of conversations it should have. For example, if a user asks the model to write an
essay about why one race is inferior or how to hijack a plane, should the model
comply?

The earliest successful preference finetuning algorithm, which is still popular today, is RLHF.
RLHF consists of two parts:<br>
1. Train a reward model that scores the foundation model’s outputs.<br>
2. Optimize the foundation model to generate responses for which the reward
model will give maximal scores.<br>

### Sampling
A model constructs its outputs through a process known as sampling. This section
discusses different sampling strategies and sampling variables, including temperature,
top-k, and top-p.

The right sampling strategy can make a model generate responses more suitable for
your application. For example, one sampling strategy can make the model generate
more creative responses, whereas another strategy can make its generations more
predictable.

#### Temperature
To redistribute the probabilities of the possible values, you can sample with a temperature.
Intuitively, a higher temperature reduces the probabilities of common tokens,
and as a result, increases the probabilities of rarer tokens. This enables models to create
more creative responses.

The higher the temperature, the less likely it is that the model is going to pick the
most obvious value.

#### Top-k
Top-k is a sampling strategy to reduce the computation workload without sacrificing
too much of the model’s response diversity.

A smaller k value makes the text more predictable but less interesting, as the model is
limited to a smaller set of likely words.

#### Top-p
In top-k sampling, the number of values considered is fixed to k. However, this number
should change depending on the situation. For example, given the prompt “Do
you like music? Answer with only yes or no.” the number of values considered should
be two: yes and no. Given the prompt “What’s the meaning of life?” the number of
values considered should be much larger.

#### min-p
A related sampling strategy is min-p, where you set the minimum probability that a
token must reach to be considered during sampling.

#### Stopping condition
An autoregressive language model generates sequences of tokens by generating one
token after another. A long output sequence takes more time, costs more compute
(money),28 and can sometimes annoy users. We might want to set a condition for the
model to stop the sequence. One easy method is to ask models to stop generating after a fixed number of tokens.

#### The Probabilistic Nature of AI
The way AI models sample their responses makes them probabilistic.

If an AI model thinks that
Vietnamese cuisine has a 70% chance of being the best cuisine in the world and Italian
cuisine has a 30% chance, it’ll answer “Vietnamese cuisine” 70% of the time and
“Italian cuisine” 30% of the time. The opposite of probabilistic is deterministic, when
the outcome can be determined without any random variation.
This probabilistic nature can cause inconsistency and hallucinations. Inconsistency is
when a model generates very different responses for the same or slightly different
prompts. Hallucination is when a model gives a response that isn’t grounded in facts.

This probabilistic nature makes AI great for creative tasks. What is creativity but the
ability to explore beyond the common paths—to think outside the box? AI is a great
sidekick for creative professionals. It can brainstorm limitless ideas and generate
never-before-seen designs. However, this same probabilistic nature can be a pain for
everything else.

#### Inconsistency
Model inconsistency manifests in two scenarios:

1. Same input, different outputs: Giving the model the same prompt twice leads to
two very different responses.<br>
2. Slightly different input, drastically different outputs: Giving the model a slightly
different prompt, such as accidentally capitalizing a letter, can lead to a very different
output.<br>

#### Hallucination
Hallucinations are fatal for tasks that depend on factuality. If you’re asking AI to help
you explain the pros and cons of a vaccine, you don’t want AI to be pseudo-scientific.

## Chapter 3: Evaluation Methodology

As teams rush to adopt AI, many quickly realize that the biggest hurdle to bringing
AI applications to reality is evaluation. For some applications, figuring out evaluation
can take up the majority of the development effort

#### Understanding Language Modeling Metrics
Most autoregressive language models are trained using
cross entropy or its relative, perplexity. When reading papers and model reports, you
might also come across bits-per-character (BPC) and bits-per-byte (BPB); both are
variations of cross entropy.

All four metrics—cross entropy, perplexity, BPC, and BPB—are closely related. If you
know the value of one, you can compute the other three, given the necessary information.

##### Entropy
Entropy measures how much information, on average, a token carries. The higher the
entropy, the more information each token carries, and the more bits are needed to
represent a token.

Intuitively, entropy measures how difficult it is to predict what comes next in a language.
The lower a language’s entropy (the less information a token of a language
carries), the more predictable that language. In

##### Cross Entropy
When you train a language model on a dataset, your goal is to get the model to learn
the distribution of this training data. In other words, your goal is to get the model to
predict what comes next in the training data. A language model’s cross entropy on a
dataset measures how difficult it is for the language model to predict what comes
next in this dataset.

##### Bits-per-Character and Bits-per-Byte
One unit of entropy and cross entropy is bits. If the cross entropy of a language
model is 6 bits, this language model needs 6 bits to represent each token.
Since different models have different tokenization methods—for example, one model
uses words as tokens and another uses characters as tokens—the number of bits per
token isn’t comparable across models. Some use the number of bits-per-character
(BPC) instead. If the number of bits per token is 6 and on average, each token consists
of 2 characters, the BPC is 6/2 = 3.

##### Perplexity
Perplexity is the exponential of entropy and cross entropy. Perplexity is often shortened
to PPL. Given a dataset with the true distribution P, its perplexity is defined as:
PPL (P) = 2^(H (P))

If cross entropy measures how difficult it is for a model to predict the next token,
perplexity measures the amount of uncertainty it has when predicting the next token.
Higher uncertainty means there are more possible options for the next token.

##### Perplexity Interpretation and Use Cases
As discussed, cross entropy, perplexity, BPC, and BPB are variations of language
models’ predictive accuracy measurements. The more accurately a model can predict
a text, the lower these metrics are.

### Similarity Measurements Against Reference Data

There are four ways to measure the similarity between two open-ended texts:<br>
1. Asking an evaluator to make the judgment whether two texts are the same<br>
2. Exact match: whether the generated response matches one of the reference
responses exactly<br>
3. Lexical similarity: how similar the generated response looks to the reference
responses<br>
4. Semantic similarity: how close the generated response is to the reference responses
in meaning (semantics)<br>
<br>

**Fuzzy Matching:** One way to measure lexical similarity is approximate string matching, known colloquially
as fuzzy matching. It measures the similarity between two texts by counting
how many edits it’d need to convert from one text to another, a number called edit
distance.
<br>

**n-gram similarity:** Another way to measure lexical similarity is n-gram similarity, measured based on
the overlapping of sequences of tokens, n-grams, instead of single tokens. A 1-gram
(unigram) is a token. A 2-gram (bigram) is a set of two tokens. “My cats scare the
mice” consists of four bigrams: “my cats”, “cats scare”, “scare the”, and “the mice”.
You measure what percentage of n-grams in reference responses is also in the generated
response.<br>

**Semantic (aka embedding) similarity**
Lexical similarity measures whether two texts look similar, not whether they have the
same meaning. Consider the two sentences “What’s up?” and “How are you?” Lexically,
they are different—there’s little overlapping in the words and letters they use.
However, semantically, they are close. Conversely, similar-looking texts can mean
very different things. “Let’s eat, grandma” and “Let’s eat grandma” mean two completely
different things.

Semantic similarity aims to compute the similarity in semantics. This first requires
transforming a text into a numerical representation, which is called an embedding.
For example, the sentence “the cat sits on a mat” might be represented using an
embedding that looks like this: [0.11, 0.02, 0.54]. Semantic similarity is, therefore,
also called embedding similarity.

The similarity
between two embeddings can be computed using metrics such as cosine similarity.
Two embeddings that are exactly the same have a similarity score of 1. Two opposite
embeddings have a similarity score of –1.

Metrics for semantic textual similarity include BERTScore (embeddings are generated
by BERT) and MoverScore (embeddings are generated by a mixture of
algorithms).

### Introduction to Embedding
Since computers work with numbers, a model needs to convert its input into numerical
representations that computers can process. An embedding is a numerical representation
that aims to capture the meaning of the original data.
An embedding is a vector. For example, the sentence “the cat sits on a mat” might be
represented using an embedding vector that looks like this: [0.11, 0.02, 0.54].
Here, I use a small vector as an example. In reality, the size of an embedding vector
(the number of elements in the embedding vector) is typically between 100 and
10,000.

Models trained especially to produce embeddings include the open source models
BERT, CLIP (Contrastive Language–Image Pre-training), and Sentence Transformers.

**Embedding sizes used by common models**<br>
Model               Embedding     size<br>
Google’s BERT =>      BERT base:     768<br>
                    BERT large:   1024<br>
OpenAI’s CLIP =>      Image:         512<br>
                    Text:          512<br>
OpenAI Embeddings API => text-embedding-3-small: 1536<br>
                      text-embedding-3-large: 3072<br>
Cohere’s Embed v3 => embed-english-v3.0: 1024<br>
                    embed-english-light-3.0: 384<br>
<br>
Because models typically require their inputs to first be transformed into vector representations,
many ML models, including GPTs and Llamas, also involve a step to
generate embeddings.

At a high level, an embedding algorithm is considered good if more-similar texts
have closer embeddings, measured by cosine similarity or related metrics.

A joint embedding space that can represent data of different modalities is a multimodal
embedding space. In a text–image joint embedding space, the embedding of an
image of a man fishing should be closer to the embedding of the text “a fisherman”
than the embedding of the text “fashion show”. This joint embedding space allows
embeddings of different modalities to be compared and combined. For example, this
enables text-based image search. Given a text, it helps you find images closest to this
text.
p136



