# BERT: Bidirectional Encoder Representations from Transformers

BERT is a language representation model built on the Transformer encoder stack. Unlike autoregressive language models that read text strictly left-to-right, BERT is trained to build deeply bidirectional representations by jointly conditioning on both left and right context in every layer.

## Architecture

BERT uses only the encoder half of the original Transformer architecture — there is no decoder and no cross-attention. Each encoder layer applies multi-head self-attention followed by a position-wise feed-forward network, with residual connections and layer normalization around each sublayer, exactly as in the standard Transformer encoder. The original paper released two sizes: BERT-base (12 layers, 768 hidden dimensions, 12 attention heads, ~110 million parameters) and BERT-large (24 layers, 1024 hidden dimensions, 16 attention heads, ~340 million parameters).

Input to BERT is a sequence of WordPiece tokens, preceded by a special [CLS] token whose final hidden state is used as an aggregate sequence representation for classification tasks, and containing a [SEP] token to separate sentence pairs. Each token's input embedding is the sum of a token embedding, a learned segment embedding (indicating which of two sentences the token belongs to), and a positional embedding.

## Masked Language Modeling

BERT cannot simply be trained with a standard left-to-right next-token objective, because bidirectional self-attention would let each position trivially see the answer through later positions. Instead, BERT is pretrained with a **masked language modeling (MLM)** objective: 15% of input tokens are randomly selected, and of those, 80% are replaced with a special [MASK] token, 10% are replaced with a random token, and 10% are left unchanged. The model is then trained to predict the original identity of every selected token using the final hidden state at that position, based on both left and right context. This masking scheme prevents the model from trivially copying the masked token forward while still exposing it to real (unmasked) tokens at training time, reducing the train/inference mismatch that would occur if [MASK] never appeared outside of training.

## Next Sentence Prediction

The original BERT pretraining also included a **next sentence prediction (NSP)** objective: given two segments A and B, the model predicts whether B is the sentence that actually follows A in the original text, or a random sentence sampled from elsewhere in the corpus. This was intended to teach the model relationships between sentence pairs, which is useful for downstream tasks like question answering and natural language inference. Later work (such as RoBERTa) found that removing NSP and training only with MLM, using larger batches and more data, did not hurt — and often improved — downstream performance, suggesting NSP was not essential to BERT's success.

## Fine-Tuning

After pretraining on a large unlabeled corpus, BERT is adapted to downstream tasks via fine-tuning: a small task-specific head (for example, a linear classification layer on top of the [CLS] token) is added, and the entire model — pretrained weights included — is fine-tuned end-to-end on labeled task data, typically for just a few epochs. This "pretrain then fine-tune" paradigm, where a single pretrained encoder can be adapted to many different downstream tasks (classification, span extraction for question answering, token tagging) with minimal architecture changes, was one of BERT's most influential contributions and became the standard recipe for transfer learning in NLP.
