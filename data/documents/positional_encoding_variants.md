# Positional Encoding Variants: Sinusoidal, Learned, RoPE, and ALiBi

Self-attention, as used in the Transformer, is permutation-invariant by construction: attention weights are computed from content alone (queries, keys, and values), with no inherent notion of token order or position. Without some explicit mechanism to inject position information, a Transformer would treat an input sequence as an unordered "bag of tokens." Positional encoding schemes exist to solve this problem, and several distinct approaches have been developed since the original Transformer.

## Sinusoidal Positional Encoding

The original Transformer paper used a fixed (non-learned) sinusoidal positional encoding, adding a position-dependent vector to each token's input embedding before the first layer. For position pos and embedding dimension index i, the encoding is defined using sine and cosine functions of different frequencies:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Because sine and cosine functions are used, and because for any fixed offset k, PE(pos+k) can be expressed as a linear function of PE(pos), the model has an easier time learning to attend based on *relative* position, not just absolute position. Since this encoding is a fixed mathematical function rather than a learned parameter, it can in principle generalize to sequence lengths longer than any seen during training, though in practice extrapolation quality is often poor without further adaptation.

## Learned Positional Embeddings

An alternative, used by BERT and GPT-2 among others, is to simply learn a distinct embedding vector for each position index, exactly like a token embedding but indexed by position instead of token identity, and add it to the token embedding. This is simpler to implement and can outperform fixed sinusoidal encodings when trained on enough data, but it has a hard limitation: the model has no defined embedding for any position beyond the maximum sequence length seen during training, so it cannot process longer sequences at inference time without extending the embedding table and further training.

## Rotary Positional Embedding (RoPE)

Rotary Positional Embedding takes a fundamentally different approach: rather than adding a positional vector to the input embedding, RoPE encodes position by rotating each query and key vector by an angle proportional to its position, within pairs of embedding dimensions treated as 2D planes. Critically, the dot product between a query at position m and a key at position n, after both have been rotated by RoPE, depends only on their *relative* position (m - n), not on their absolute positions. This gives RoPE the relative-position sensitivity that made the original sinusoidal encoding appealing, while injecting position information directly into the attention computation rather than into the input embeddings, which several large language models (including LLaMA and its successors) have found to generalize better to sequence lengths beyond those seen in training.

## ALiBi: Attention with Linear Biases

ALiBi takes yet another approach, dispensing with positional embeddings entirely. Instead, it adds a fixed, non-learned penalty directly to the attention scores before the softmax, proportional to the distance between the query and key positions: the further apart two positions are, the larger the negative bias subtracted from their attention score, discouraging (but not strictly preventing) attention to distant tokens. Each attention head uses a different fixed slope for this linear penalty. Because this bias is a simple linear function of relative distance rather than a fixed-size table of position embeddings, ALiBi has been shown to extrapolate to much longer sequences at inference time than were seen during training, with substantially better length generalization than either sinusoidal or learned absolute positional embeddings.

## Why This Matters for Long-Context Models

The choice of positional encoding scheme has become increasingly consequential as language models are pushed to handle longer and longer context windows. Absolute positional schemes (learned embeddings, and to a lesser extent the original sinusoidal encoding) tend to degrade sharply beyond their training length, while relative schemes like RoPE and ALiBi were specifically designed to make length extrapolation and long-context generalization more robust, which is part of why they have become the default choice in most modern large language model architectures.
