# Recurrent Neural Networks

Recurrent neural networks (RNNs) are a class of neural network designed for sequential data, where the output at each time step depends on both the current input and a hidden state carried forward from previous time steps. This makes them a natural fit for tasks like language modeling, time-series forecasting, and speech recognition, where order and context matter.

## The Recurrence

At each time step t, a standard (vanilla) RNN computes a new hidden state h_t from the current input x_t and the previous hidden state h_{t-1}:

h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)

The same weight matrices W_xh and W_hh are reused at every time step, which is what makes the network "recurrent" — parameters are shared across the sequence rather than learned separately for each position. An output y_t can then be computed from h_t via a separate output weight matrix, y_t = W_hy * h_t + b_y.

Because the hidden state is updated sequentially, RNNs process sequences one token at a time, unlike architectures such as the Transformer that can process all positions in parallel. This sequential dependency is the main reason RNNs are slower to train on long sequences on modern parallel hardware.

## Backpropagation Through Time

RNNs are trained using backpropagation through time (BPTT), which unrolls the recurrence across all time steps and applies the standard backpropagation algorithm to the unrolled computational graph. Gradients with respect to the shared weight matrices are accumulated across every time step, since the same weights are reused at each position.

## Vanishing and Exploding Gradients

The central weakness of vanilla RNNs is the vanishing and exploding gradient problem. Because the same weight matrix W_hh is multiplied repeatedly through the chain rule across many time steps, gradients can shrink toward zero (vanish) or grow without bound (explode) as they are backpropagated through a long sequence. Vanishing gradients make it difficult for the network to learn dependencies between distant time steps, since the gradient signal from a later error barely reaches earlier hidden states. Exploding gradients cause unstable, oscillating updates and can be mitigated with gradient clipping, which rescales the gradient vector when its norm exceeds a threshold.

This vanishing-gradient limitation is the primary motivation behind gated architectures such as LSTM and GRU, which introduce explicit gating mechanisms to control how information flows and decays across time steps.

## Bidirectional RNNs

A bidirectional RNN runs two separate hidden-state recurrences over the same input sequence: one processing the sequence forward (left to right) and one processing it backward (right to left). The two hidden states at each position are typically concatenated before being passed to the output layer. This allows the model's representation at any given position to incorporate context from both earlier and later parts of the sequence, which is useful for tasks like named entity recognition or part-of-speech tagging where future context disambiguates the current token.

## Limitations Relative to Attention-Based Models

Because RNNs compress all preceding context into a single fixed-size hidden state vector, they can struggle to retain information over very long sequences — a bottleneck often described as compressing an entire history into one vector. Attention mechanisms, and the Transformer architecture built around self-attention, address this by allowing every position to directly attend to every other position in the sequence, rather than passing information through a chain of hidden states. This removes the fixed-size bottleneck and allows constant-length paths between any two positions, which is one of the key reasons Transformers have displaced RNNs in many large-scale sequence modeling tasks.
