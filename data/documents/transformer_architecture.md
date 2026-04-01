# The Transformer Architecture

## Introduction

The Transformer is a deep learning architecture introduced by Vaswani et al. in the 2017
paper "Attention Is All You Need." It was originally designed for sequence-to-sequence tasks
such as machine translation, but has since become the dominant architecture in natural
language processing and beyond. Unlike recurrent neural networks (RNNs), the Transformer
processes all tokens in a sequence simultaneously rather than one at a time, which enables
much more efficient training on modern hardware.

The core innovation of the Transformer is the **self-attention mechanism**, which allows
every token in a sequence to attend to every other token in a single operation. This gives
the model a direct path to learn long-range dependencies without the vanishing gradient
problems that plague deep RNNs.

## Self-Attention

Self-attention computes a weighted sum of value vectors, where the weights are determined
by the compatibility between query and key vectors. Given an input sequence of embeddings,
three learned linear projections produce a matrix of queries Q, keys K, and values V.
The attention score for each pair of positions is computed as the dot product of the
corresponding query and key vectors, scaled by the square root of the key dimension:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

The scaling factor `1 / sqrt(d_k)` prevents the dot products from growing too large in
high-dimensional spaces, which would push the softmax into regions with very small gradients.

Each token produces its own query vector, representing what it is "looking for," and a key
vector, representing what information it "offers" to other tokens. The dot product of a
query with all keys produces unnormalized attention logits; after softmax, these become
attention weights that sum to one. The output for each token is the weighted average of
all value vectors.

## Multi-Head Attention

Rather than performing a single attention operation, the Transformer uses **multi-head
attention**, which runs several attention operations in parallel over learned linear
projections of the queries, keys, and values. With h heads and model dimension d_model,
each head operates in a subspace of dimension d_k = d_v = d_model / h.

The outputs of all heads are concatenated and projected back to d_model:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)

Multi-head attention allows the model to jointly attend to information from different
representation subspaces at different positions. A single attention head is limited to
attending to a weighted average; multiple heads allow the model to attend to several
distinct patterns simultaneously.

## Positional Encoding

Because the Transformer has no recurrence or convolution, it has no inherent notion of
token order. To inject sequence position information, the original Transformer adds a
**positional encoding** to the input embeddings before the first layer.

The original paper used fixed sinusoidal encodings:

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

These encodings have the property that the encoding for position `pos + k` can be
expressed as a linear function of the encoding for position `pos`, which allows the model
to learn to attend by relative position. Later work replaced fixed encodings with learned
positional embeddings (e.g., BERT) or relative position encodings (e.g., RoPE, ALiBi)
that generalize better to sequences longer than those seen during training.

## Encoder and Decoder Stacks

The original Transformer is an **encoder-decoder** model. Both stacks are composed of
repeated identical layers.

### Encoder Layer

Each encoder layer has two sublayers:
1. **Multi-head self-attention**: every token attends to every other token in the input.
2. **Feed-forward network (FFN)**: a two-layer MLP applied independently to each position.

Both sublayers are wrapped in a residual connection followed by layer normalisation:

    output = LayerNorm(x + Sublayer(x))

This pre-norm or post-norm design (the original uses post-norm) stabilizes training in
deep stacks by keeping gradient magnitudes well-behaved.

### Decoder Layer

Each decoder layer adds a third sublayer between self-attention and FFN:
- **Masked self-attention**: the decoder can only attend to previous positions (causal masking),
  enforcing autoregressive generation.
- **Cross-attention**: each decoder token attends over the full encoder output, giving the
  decoder access to the entire source sequence at every generation step.

## Feed-Forward Networks

The feed-forward sublayer within each Transformer layer is a two-layer fully-connected
network with a non-linear activation in between:

    FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2

The original paper used ReLU; modern models typically use GELU or SwiGLU. The inner
dimension is usually 4x the model dimension (e.g., d_ff = 4 * d_model = 2048 for a
512-dimensional model), giving the FFN substantial capacity to store factual knowledge.

## Encoder-Only and Decoder-Only Variants

Modern practice has split the original encoder-decoder design into specialised variants:

- **Encoder-only models** (e.g., BERT, RoBERTa): use only the encoder stack with
  bidirectional attention. Trained with masked language modelling. Best suited for
  understanding tasks: classification, NER, question answering over a short context.

- **Decoder-only models** (e.g., GPT, LLaMA, Claude): use only the decoder stack with
  causal masking. Trained with next-token prediction. Naturally suited for generation
  and have become the dominant paradigm for large language models.

- **Encoder-decoder models** (e.g., T5, BART): retain the original design. Excel at
  tasks that require mapping one sequence to another, such as summarization and
  translation.

## Scaling Laws

Research by Kaplan et al. (2020) established **neural scaling laws**: model performance
on language modelling improves predictably as a power law with increases in model size,
dataset size, and compute budget. The key finding is that compute-optimal training
allocates roughly equal resources to model size and data tokens.

Hoffmann et al. (2022), the Chinchilla paper, refined these findings: for a given compute
budget, previous large models were significantly under-trained relative to their size.
The Chinchilla-optimal recipe is approximately 20 tokens of training data per model
parameter. This insight drove a shift toward training smaller models on much larger
datasets, producing models that are both more capable and cheaper to serve.

## Key Hyperparameters

| Hyperparameter | Typical range | Effect |
|---|---|---|
| d_model | 512 – 12288 | Embedding and residual stream width |
| n_heads | 8 – 96 | Number of attention heads |
| n_layers | 6 – 96 | Depth of encoder or decoder stack |
| d_ff | 2048 – 49152 | FFN inner dimension (usually 4× d_model) |
| Context length | 512 – 1M+ tokens | Maximum sequence length |
| Dropout | 0.0 – 0.3 | Regularisation during training |

Larger models generally use larger values across all dimensions, but the ratios between
them (e.g., d_ff / d_model ≈ 4, d_k = d_model / n_heads) are often kept roughly constant.
