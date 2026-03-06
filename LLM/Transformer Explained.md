# Transformers, the tech behind LLMs
https://www.3blue1brown.com/lessons/gpt

https://www.youtube.com/watch?v=wjZofJX0v4M

## Embeddings
The model has a predefined vocabulary, some list of all possible words, say 50,000 of them. The first matrix of the transformer, known as the embedding matrix, will have one column for each of these words. These columns determine what vector each word turns into in that first step. 

![Alt](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2024/gpt/embedding.png)

Visualizing a list of three numbers as coordinates for a point in 3D space is no problem, but word embeddings tend to be very high dimensional. For GPT-3, they have 12,288 dimensions, and as you will see, it matters to work in a space with lots of distinct directions.

## Directions
The big idea we need to understand here is that as a model tweaks and tunes its weights to decide how exactly words get embedded as vectors during training, it tends to settle on a set of embeddings where directions in this space have meaning. Below, a simple word-to-vector model is running, and when I run a search for all words whose embeddings are closest to that of tower, they all generally have the same vibe.
![Alt](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2024/gpt/gender.png)

## Dot Product
![Alt](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2024/gpt/HowToDot.png)
One bit of mathematical intuition helpful to have in mind as we continue is how the dot product of two vectors can be thought of as measuring how well they align. Computationally, dot products involve multiplying all aligning components and adding the result. Geometrically, the dot product is positive when the vectors point in a similar direction, zero if they're perpendicular, and negative when they point in opposite directions.

## Beyond Words - Context
Again, how specifically each word gets embedded is learned using data. The embedding matrix, whose columns store the embedding of each word, is the first pile of weights in our model. Using the GPT-3 numbers, the vocabulary size is 50,257, and again, technically this consists not of words, per se, but different little chunks of text called tokens. The embedding dimension is 12,288, giving us 617,558,016 weights in total for this first step. Let’s go add that to a running tally, remembering that by the end we should count up to 175 billion weights.

For example, a vector that started its life as the embedding of the word “king” may progressively get tugged and pulled by the various blocks in the network to end up pointing in a much more nuanced direction that somehow encodes a king who lived in Scotland, who had achieved his post after murdering the previous king, who is being described in Shakespearean language, and so on.

Think about our understanding of a word, like quill. Its meaning is clearly informed by its surroundings and context, whether it be a hedgehog quill or a type of pen. Sometimes, we may even include context from a long distance away. When putting together a model that is able to predict the next word, the goal is to somehow empower it to do the same thing: take in context efficiently.

## When we start:
In that very first step, when you create the array of vectors based on the input text, each one is simply plucked out of this embedding matrix, and each one only encodes the meaning of the single word it's associated with. It's effectively a lookup table with no input from the surroundings. But as these vectors flow through the network, we should view the primary goal of this network as enabling each one of those vectors to soak up meaning that is more rich and specific than what mere individual words can represent.

## Context Window
The network can only look at a fixed number of vectors at a time, known as its context size. GPT-3 was trained with a context size of 2048 tokens. So the data flowing through the network will look like this array of 2048 columns, each of which has around 12k dimensions. This context size limits how much text the transformer can incorporate to make its prediction of the next word, which is why long conversations with the early versions of ChatGPT often gave the feeling of the bot losing the thread of conversation.

## Unembedding
We'll go into the details of the Attention Block in the next chapter, but first we'll skip ahead and talk about what happens at the very end of the Transformer. 

This process involves two steps. The first is to use another matrix **[Unembedding Matrix]** that maps the very last vector in the context to a list of ~50,000 values, one for each token in the vocabulary, then there’s a function that normalizes this into a probability distribution, called softmax.

This is often called the unembedding matrix, Again, like all the weight matrices we’ll see, its entries begin random but are tuned during the training process.
​
 Again, like all the weight matrices we’ll see, its entries begin random but are tuned during the training process.

 Keeping score on our total number of parameters, the unembedding matrix has one row for each word in the vocabulary, giving 50,257 words, and each row has the same number of elements as the dimension of the embedding, giving 12,288 columns. It’s very similar to the embedding matrix, just with the dimensions of the rows and columns swapped, so it adds another 617M parameters to the network, making our parameter count so far a little over a billion; a small but not insignificant fraction of the 175 billion that we'll end up with in total.

## Softmax
The idea is that if we want a sequence of numbers to serve as a probability distribution, say a distribution over all possible next words, all the values should be between 0 and 1 and should all add up to be 1. However, in deep learning, where so much of what we do looks like a matrix-vector products, the outputs we get by default won’t abide by this at all. The values are often negative or sometimes greater than 1, and they almost certainly don’t all add up to 1.

Softmax turns an arbitrary list of numbers into a valid distribution, in such a way that the largest values end up closest to 1, and the smaller values end up closer to 0.

The reason for calling it softmax is that instead of simply pulling out the biggest value, it produces a distribution that gives weight to all the relatively large values, commensurate with how large they are. If one entry in the input is much bigger than the rest, the corresponding output will be very close to 1, so sampling from the distribution is likely the same as just choosing the maximizing index from the input.

![Alt](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2024/gpt/howsoftmax.png)

## Extra
In some cases, such as when ChatGPT is using this distribution to sample each next word, we can add a little extra spice into this softmax function with a constant T inserted into all these exponents. We call it "temperature" since it vaguely resembles the role of temperature in certain thermodynamics equations. The effect of this constant is that when T is larger, more weight is given to lower values, making the distribution more uniform. When T is smaller, the larger values will dominate the distribution, and in the extreme setting when T is equal to 0, all the weight goes to the maximum value.

![Alt](https://3b1b-posts.us-east-1.linodeobjects.com/content/lessons/2024/gpt/temperatureReal.png)

## Logits
In the same way that we might call the components of the output of this function probabilities, people often refer to the components of the input as logits. So when you feed in some text, have the word embeddings flow through the network, and do this final multiplication by the unembedding matrix, machine learning people would refer to the components of that raw unnormalized output as the logits for the next word prediction.

## RECAP
The goal of the model that we began studying in the last chapter is to take in a piece of text and predict what will come next.

The input text that is taken in is first broken up into little pieces that we call tokens, which are often words or pieces of words. This process, known as tokenization, is a preprocessing step that is done before any of the input actually enters the transformer.

To make the examples in this lesson simpler for us to conceptualize, let's pretend that these tokens are always just words.

The tokens then enter the transformer, where the first step is to associate each token with a high-dimensional vector, which we call its embedding.

The important idea that we should have in mind as we continue into the attention mechanism is that directions in this high-dimensional space of all possible embeddings can correspond with meaning.

The aim of a transformer is to progressively adjust these embeddings so that they don't merely encode data of an individual word, but rather a much richer contextual meaning.


# Attention in transformers, step-by-step
https://www.3blue1brown.com/lessons/attention#title

https://www.youtube.com/watch?v=eMlx5fFNoYc




