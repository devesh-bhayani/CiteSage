# LSTM and GRU: Gated Recurrent Architectures

Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU) are extensions of the vanilla recurrent neural network designed specifically to address the vanishing gradient problem, allowing gradients — and therefore learned dependencies — to propagate across much longer sequences.

## LSTM Cell Structure

An LSTM cell maintains two separate state vectors passed between time steps: a hidden state h_t and a cell state c_t. The cell state acts as a memory conveyor belt that information can flow along largely unchanged, modified only by targeted additions and removals controlled by three gates.

The **forget gate** decides what fraction of the previous cell state to discard:
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)

The **input gate** decides what new information to write into the cell state:
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
c-tilde_t = tanh(W_c * [h_{t-1}, x_t] + b_c)

The cell state is then updated as:
c_t = f_t * c_{t-1} + i_t * c-tilde_t

The **output gate** decides what part of the cell state to expose as the hidden state:
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)

Each gate is a sigmoid-activated layer producing values between 0 and 1, acting as a learned, per-dimension switch. Because the cell state update is largely additive (f_t * c_{t-1} + i_t * c-tilde_t) rather than repeatedly multiplied through a single weight matrix, gradients can flow through many time steps without vanishing as quickly as in a vanilla RNN — this additive path is often called the "constant error carousel."

## GRU: A Simplified Alternative

The Gated Recurrent Unit simplifies the LSTM by merging the cell state and hidden state into a single vector and reducing the gating mechanism to two gates instead of three.

The **update gate** z_t controls the interpolation between the previous hidden state and a candidate new hidden state:
z_t = sigmoid(W_z * [h_{t-1}, x_t])

The **reset gate** r_t controls how much of the previous hidden state is used when computing the candidate:
r_t = sigmoid(W_r * [h_{t-1}, x_t])
h-tilde_t = tanh(W * [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h-tilde_t

GRUs have fewer parameters than LSTMs because they lack a separate output gate and cell state, which makes them faster to train and often competitive with LSTMs on many tasks, though LSTMs can retain a slight edge on tasks requiring very long-range memory due to the extra capacity of the separate cell state.

## Practical Use and Decline

LSTMs and GRUs were the dominant architecture for sequence modeling — machine translation, speech recognition, language modeling — through the mid-2010s. They remain useful for smaller-scale sequential tasks and settings with strict latency or memory constraints, since inference is a single sequential pass with modest memory footprint. However, for large-scale language modeling and machine translation, they have largely been superseded by the Transformer architecture, which replaces the sequential recurrence with self-attention, enabling full parallelization across sequence positions during training and better modeling of very long-range dependencies without the sequential bottleneck.
