# GPT and Autoregressive Language Modeling

GPT (Generative Pre-trained Transformer) refers to a family of language models built on the decoder-only half of the Transformer architecture, trained to predict the next token in a sequence given all previous tokens. This autoregressive framing — factoring the probability of a sequence as a product of conditional next-token probabilities — is the foundation of modern large language models.

## Decoder-Only Architecture

Unlike BERT, which uses the bidirectional Transformer encoder, GPT uses a stack of Transformer decoder blocks, each consisting of masked (causal) multi-head self-attention followed by a position-wise feed-forward network, with residual connections and layer normalization around each sublayer. There is no encoder and no cross-attention sublayer, since GPT is not conditioning generation on a separate source sequence the way the original encoder-decoder Transformer was for machine translation.

The defining feature is the **causal attention mask**: at position i, the self-attention mechanism is only allowed to attend to positions 1 through i, not to any future position. This is enforced by adding negative infinity to the attention scores for all disallowed (future) positions before the softmax, which drives their attention weight to zero. This causal masking is what allows GPT to be trained efficiently in parallel across an entire sequence with teacher forcing, while still guaranteeing that at inference time, generating token i only ever depended on tokens before it — exactly matching how the model must generate text autoregressively, one token at a time, at inference.

## Training Objective

GPT is trained to maximize the log-likelihood of the training corpus under the autoregressive factorization:

log P(x_1, ..., x_n) = sum_{t=1}^{n} log P(x_t | x_1, ..., x_{t-1})

In practice this means, at each position, the model outputs a probability distribution over the vocabulary for the next token, and the loss is the cross-entropy between that distribution and the actual next token in the training data. This is often called "next-token prediction" and requires no labeled data — the supervision signal is simply the next token in a large corpus of raw text, which is why this style of pretraining scales so easily with more unlabeled text.

## Scaling and In-Context Learning

Successive GPT models (GPT-2, GPT-3, and beyond) scaled the same decoder-only architecture to progressively larger parameter counts and training corpora, and this scaling revealed **in-context learning**: sufficiently large models can perform a task described or demonstrated only in the prompt, without any gradient updates, purely by conditioning on examples provided at inference time. This is distinct from the fine-tuning paradigm used by BERT, where task adaptation requires updating model weights on labeled data — with in-context learning, the model's weights are frozen and the "adaptation" happens entirely through the input context.

## Sampling at Inference Time

Because GPT defines a probability distribution over the next token at each step, generating text requires a decoding strategy. Greedy decoding always picks the single highest-probability token, which is deterministic but can produce repetitive or degenerate text. Sampling strategies such as temperature scaling (dividing logits by a temperature before the softmax to sharpen or flatten the distribution), top-k sampling (restricting sampling to the k highest-probability tokens), and top-p / nucleus sampling (restricting to the smallest set of tokens whose cumulative probability exceeds p) are commonly used to trade off diversity against coherence in generated text. Setting temperature to zero effectively reduces sampling to greedy decoding, which is why temperature=0 is commonly used when deterministic, reproducible output is required, such as in automated evaluation pipelines.
