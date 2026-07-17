# Vision Transformer (ViT)

The Vision Transformer (ViT) applies the standard Transformer encoder architecture — originally developed for text — directly to image classification, with minimal changes, by reformulating an image as a sequence of patches rather than a sequence of tokens.

## From Image to Sequence: Patch Embeddings

Convolutional neural networks process images via local convolutional filters that slide across the spatial grid, building up larger receptive fields layer by layer. ViT instead converts an image into something a standard Transformer encoder can consume directly: the image is split into a grid of fixed-size, non-overlapping square patches (commonly 16x16 pixels), each patch is flattened into a single vector, and that vector is linearly projected into the model's embedding dimension — exactly analogous to how a word embedding lookup converts a discrete token into a vector. A 224x224 pixel image split into 16x16 patches yields 196 patches, each treated as one "token" in the resulting sequence.

Because self-attention has no inherent notion of spatial position (just as it has none for sequence position in text), a learned positional embedding is added to each patch embedding, encoding where in the image grid that patch was located. An extra learnable [CLS] token (borrowed directly from BERT's classification-token convention) is prepended to the sequence of patch embeddings, and its final hidden state after passing through the Transformer encoder is used as the aggregate image representation for classification.

## Encoder Architecture

Once an image has been converted into a sequence of patch embeddings plus a [CLS] token, ViT applies a standard Transformer encoder stack — the same multi-head self-attention plus position-wise feed-forward sublayers, with residual connections and layer normalization, used in text encoder models like BERT. No image-specific inductive biases (such as convolutional weight sharing or spatial locality) are built into the architecture at all; every patch can attend to every other patch from the very first layer, regardless of spatial distance between them.

## The Inductive Bias Trade-off

Convolutional neural networks have strong built-in inductive biases suited to images: translation equivariance (a shifted input produces a correspondingly shifted output) and locality (early layers only look at small neighborhoods of pixels), both of which are useful priors for natural images where nearby pixels are typically far more related than distant ones. ViT discards these image-specific priors almost entirely, treating the image as a generic sequence problem. The consequence is that ViT tends to underperform comparably-sized CNNs when trained only on smaller datasets, since it must learn spatial relationships (which a CNN gets "for free" from its architecture) purely from data. However, when pretrained on very large image datasets, ViT has been shown to match or exceed state-of-the-art CNN performance, suggesting that with enough data, the flexibility of a less-constrained architecture can outweigh the benefit of hand-designed image-specific priors — mirroring a similar pattern seen when Transformers overtook RNNs in NLP once sufficient training data and compute became available.

## Extensions

Follow-up work has explored hybrid architectures that combine convolutional feature extraction with Transformer encoders, hierarchical vision Transformers that build multi-scale representations more similar to a CNN's feature pyramid, and self-supervised pretraining objectives for ViT (such as masked patch prediction, directly analogous to BERT's masked language modeling) that reduce the model's dependence on very large labeled datasets.
