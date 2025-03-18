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


