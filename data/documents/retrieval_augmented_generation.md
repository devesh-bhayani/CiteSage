# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is an architecture pattern for language model systems that combines a parametric language model with a non-parametric retrieval step over an external document corpus, so that generation can be grounded in retrieved evidence rather than relying solely on knowledge encoded in the model's weights during pretraining.

## Motivation

A language model's weights encode a compressed, implicit representation of whatever it was trained on, fixed at the time training finished. This creates two persistent problems: the model's knowledge is frozen at its training cutoff and cannot reflect new or updated information without retraining, and the model has no mechanism to point to a specific source justifying any particular claim, which makes fabricated (hallucinated) statements indistinguishable from accurate ones without external verification. RAG addresses both problems by retrieving relevant passages from an external, updatable corpus at query time, and conditioning the language model's generation on that retrieved text, so the corpus — not just the frozen weights — becomes a source of both current information and verifiable grounding.

## The Two-Stage Pipeline

A RAG system consists of a retrieval stage and a generation stage, run in sequence for each query. In the retrieval stage, the user's query is used to search a corpus of documents — commonly split into smaller chunks and indexed by dense vector embeddings, sparse keyword statistics (such as BM25), or both — and the top-scoring chunks are selected as candidate context. In the generation stage, the retrieved chunks are inserted into the language model's prompt alongside the original query, and the model is asked to generate an answer conditioned on that provided context, ideally citing which retrieved chunk supports each claim.

## Dense, Sparse, and Hybrid Retrieval

Dense retrieval encodes both the query and each document chunk into fixed-size embedding vectors (typically using a sentence-embedding model), and ranks chunks by vector similarity — commonly cosine similarity or dot product — between the query embedding and each chunk embedding. This captures semantic similarity even when the query and the relevant passage don't share exact wording. Sparse retrieval methods such as BM25 instead rank documents based on term-frequency statistics: how often query terms appear in a document, weighted to favor rare, discriminating terms over common ones, and normalized for document length. Sparse retrieval tends to excel at exact keyword and terminology matches (product codes, proper nouns, precise technical terms) that dense embeddings can sometimes miss or under-weight, since not all fine-grained lexical distinctions are preserved by a fixed-size embedding vector.

Because dense and sparse retrieval have complementary strengths and failure modes, many production RAG systems use **hybrid retrieval**, running both methods and combining their ranked result lists — commonly via Reciprocal Rank Fusion, which scores each document by summing 1/(k + rank) across every ranked list it appears in, favoring documents that rank well across multiple retrieval methods over documents that rank well in only one.

## Reranking

Because a fast retrieval method (dense or sparse) is usually run over the entire corpus to find a broad initial candidate set, that same fast method is not always well-suited to precisely ordering the top handful of results. Many RAG pipelines add a **reranking** stage: a heavier, more accurate but slower model — typically a cross-encoder, which processes the query and a candidate document jointly (rather than encoding them into independent vectors, as dual-encoder dense retrieval does) — re-scores only the top-N candidates from the initial retrieval step and reorders them before passing the final, smaller set to the generation stage. This two-stage retrieve-then-rerank design trades a small amount of extra latency for meaningfully more accurate ranking of the small set of chunks that actually reach the language model's context window.

## Grounding and Citation

A common failure mode in naive RAG systems is that the language model generates an answer that ignores or contradicts the retrieved context, falling back on its own (potentially outdated or fabricated) parametric knowledge instead. More rigorous RAG designs address this with explicit citation requirements (prompting the model to tag each claim with the specific source chunk it came from) combined with a verification step that checks whether the cited source text actually supports the claim it's attached to, and can decline to answer entirely when no retrieved chunk provides sufficient support for a confident response — trading some answer coverage for a stronger guarantee that every answer given is traceable to real, provided evidence.
