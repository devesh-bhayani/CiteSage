# Normalization Layers: BatchNorm and LayerNorm

Normalization layers are a core building block of modern deep networks, used to stabilize and accelerate training by controlling the scale and distribution of activations as they flow through the network.

## The Internal Covariate Shift Problem

As a deep network trains, the distribution of inputs to each layer keeps shifting as the parameters of all preceding layers are updated — a phenomenon originally described as internal covariate shift. This forces each layer to continuously adapt to a moving target distribution of inputs, which can slow training and require careful, small learning rates to avoid instability, particularly in very deep networks.

## Batch Normalization

Batch Normalization (BatchNorm) normalizes each activation across the mini-batch dimension: for a given feature, it computes the mean and variance of that feature's values across all examples in the current mini-batch, then normalizes each example's value for that feature to zero mean and unit variance:

x-hat_i = (x_i - mu_batch) / sqrt(sigma_batch^2 + epsilon)

After normalizing, BatchNorm applies a learned affine transformation, y_i = gamma * x-hat_i + beta, where gamma and beta are learnable parameters that allow the network to undo the normalization if that turns out to be optimal for a given layer. Because batch statistics (mu_batch, sigma_batch) depend on the other examples in the mini-batch, BatchNorm behaves differently at training time (using current-batch statistics) versus inference time (using a running average of batch statistics accumulated during training, so a single test example can be normalized independent of any batch it's part of).

BatchNorm's main drawback is its dependence on batch composition and size: with very small batch sizes, the batch statistics become noisy and unreliable estimates of the true population statistics, degrading performance. It is also awkward to apply to sequence models with variable-length inputs, since padding tokens can distort the batch statistics for later positions in the sequence.

## Layer Normalization

Layer Normalization (LayerNorm) sidesteps the batch-dependence problem by normalizing across the feature dimension instead of the batch dimension: for a single example, it computes the mean and variance across all the features within that one example (for instance, across the hidden dimension at a single sequence position), and normalizes using those per-example statistics:

x-hat = (x - mu) / sqrt(sigma^2 + epsilon)

where mu and sigma^2 are computed over the features of a single example, not across the batch. This makes LayerNorm's computation completely independent of batch size and of other examples in the batch, which is exactly why it is the standard choice in Transformer architectures — it applies uniformly regardless of sequence length, batch size, or padding, and behaves identically at training and inference time since there is no batch statistic to track a running average of.

## Placement: Post-Norm vs. Pre-Norm

In the original Transformer, LayerNorm is applied *after* each sublayer's output is added to its residual connection — a configuration often called "post-norm": output = LayerNorm(x + Sublayer(x)). Later work found that applying LayerNorm *before* each sublayer instead — "pre-norm": output = x + Sublayer(LayerNorm(x)) — produces more stable gradients early in training and often removes the need for the careful learning-rate warmup that post-norm Transformers typically require, because the residual path in pre-norm remains a clean, unnormalized identity connection all the way from input to output. Most large-scale Transformer language models trained since have adopted the pre-norm configuration for this training-stability benefit, even though the original architecture used post-norm.
