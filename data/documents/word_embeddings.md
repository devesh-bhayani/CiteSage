# Word Embeddings: word2vec and GloVe

Word embeddings are dense, low-dimensional vector representations of words, learned such that words with similar meanings or usage patterns end up close together in the vector space. They replaced older one-hot or sparse count-based representations, which treated every word as an independent symbol with no notion of similarity between, for example, "cat" and "dog."

## The Distributional Hypothesis

Word embedding methods are built on the distributional hypothesis: words that occur in similar contexts tend to have similar meanings. Practically, this means a word's meaning can be approximated by statistics about the other words that co-occur with it across a large text corpus, rather than by any explicit definition.

## word2vec

word2vec introduced two related training architectures for learning word embeddings efficiently from raw text, using a shallow neural network with a single hidden layer (the embedding matrix itself).

**Continuous Bag of Words (CBOW)** predicts a target word from the average of its surrounding context words within a fixed window. **Skip-gram** does the reverse: given a single target word, it predicts each of the surrounding context words independently. Skip-gram tends to perform better on rare words since it generates more training pairs per occurrence of a rare word, while CBOW is generally faster to train.

Directly computing a full softmax over the entire vocabulary at every training step is prohibitively expensive for vocabularies with hundreds of thousands of words. word2vec addresses this with **negative sampling**: instead of updating the full output layer, each training step samples a small number of "negative" words that did not actually appear in the context, and the model is trained to distinguish true context words from these negative samples using a binary logistic loss. This reduces the per-step computation from O(vocabulary size) to O(number of negative samples), making training tractable on large corpora.

A widely cited property of the resulting embeddings is that vector arithmetic captures analogical relationships, such as vector("king") - vector("man") + vector("woman") landing close to vector("queen") — evidence that the learned space encodes meaningful linear structure, not just similarity clustering.

## GloVe: Global Vectors

GloVe (Global Vectors for Word Representation) takes a different approach: rather than training on local context windows one example at a time like word2vec, GloVe first constructs a global word-word co-occurrence matrix over the entire corpus, counting how often each pair of words appears within a context window across all of the text. It then factorizes this co-occurrence statistic directly, training embeddings such that the dot product of two word vectors approximates the logarithm of their co-occurrence probability ratio. This ties the embeddings explicitly to global corpus statistics, whereas word2vec's objective only ever sees local context windows during training, without directly modeling aggregate co-occurrence counts.

In practice, word2vec and GloVe embeddings tend to perform comparably on standard word-similarity and analogy benchmarks, and the choice between them is often driven by engineering convenience rather than a large quality gap.

## Static vs. Contextual Embeddings

A key limitation of both word2vec and GloVe is that they produce **static** embeddings: each word has exactly one vector, regardless of context, so a polysemous word like "bank" (a financial institution vs. a riverbank) receives a single blended representation covering all of its senses. This is the central distinction from **contextual embeddings** produced by models like BERT and GPT, where a word's vector representation is computed dynamically from the Transformer's self-attention over the surrounding sentence, so the same word can receive a different embedding depending on the sentence it appears in.
