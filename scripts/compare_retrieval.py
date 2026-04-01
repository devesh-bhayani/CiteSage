"""Hybrid vs BM25-only vs vector-only retrieval comparison.

Runs 10 test queries against the indexed transformer_architecture.md document
and prints a side-by-side comparison of which chunks each method found,
together with reranker scores.

Run from the project root:
    uv run python scripts/compare_retrieval.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from citesage.config import get_settings
from citesage.ingestion.models import Chunk
from citesage.ingestion.storage import BM25Index, ChromaStore
from citesage.retrieval.reranker import Reranker
from citesage.retrieval.rrf import rrf_fuse

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Exact-term queries
    "Vaswani et al.",
    "QK^T",
    "d_model / h",
    "10000",
    "Chinchilla",
    # Semantic queries
    "how does attention work?",
    "what prevents large gradients?",
    "decoder generation",
    # Vague queries
    "what paper introduced this?",
    "architecture comparison",
]


# ---------------------------------------------------------------------------
# Helper: build ranked lists without reranking
# ---------------------------------------------------------------------------


def run_bm25_only(
    bm25: BM25Index,
    query: str,
    top_k: int = 20,
) -> list[Chunk]:
    raw = bm25.search(query, top_k=top_k)
    return [c for c, _ in raw]


def run_vector_only(
    store: ChromaStore,
    query: str,
    top_k: int = 20,
) -> list[Chunk]:
    raw = store.query(query_text=query, top_k=top_k)
    return [c for c, _ in raw]


def run_hybrid(
    bm25: BM25Index,
    store: ChromaStore,
    reranker: Reranker,
    query: str,
    bm25_top_k: int = 20,
    vector_top_k: int = 20,
    rerank_candidates: int = 15,
    rerank_top_k: int = 5,
) -> tuple[list[tuple[Chunk, float]], list[str], list[str]]:
    """Return (scored_chunks, bm25_ids, vector_ids) for the hybrid path."""
    bm25_raw = bm25.search(query, top_k=bm25_top_k)
    vector_raw = store.query(query_text=query, top_k=vector_top_k)

    bm25_list: list[tuple[str, Chunk]] = [(c.chunk_id, c) for c, _ in bm25_raw]
    vector_list: list[tuple[str, Chunk]] = [(c.chunk_id, c) for c, _ in vector_raw]

    bm25_ids = [cid for cid, _ in bm25_list]
    vector_ids = [cid for cid, _ in vector_list]

    fused = rrf_fuse([bm25_list, vector_list])
    candidates: list[Chunk] = [chunk for _, chunk, _ in fused[:rerank_candidates]]

    scored = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=rerank_top_k,
        skip_threshold=True,
    )
    return [(sc.chunk, sc.score) for sc in scored], bm25_ids, vector_ids


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


def _short_id(chunk_id: str) -> str:
    return chunk_id[:12]


def _preview(content: str, width: int = 60) -> str:
    one_line = " ".join(content.split())
    return (one_line[:width] + "…") if len(one_line) > width else one_line


def compare_query(
    query: str,
    bm25: BM25Index,
    store: ChromaStore,
    reranker: Reranker,
) -> dict:
    """Run all three methods and return comparison data."""
    settings = get_settings()

    # BM25-only → rerank to get scores
    bm25_chunks = run_bm25_only(bm25, query, top_k=settings.retrieval.bm25_top_k)
    bm25_scored = reranker.rerank(
        query=query,
        candidates=bm25_chunks[: settings.retrieval.rerank_candidates],
        top_k=settings.retrieval.rerank_top_k,
        skip_threshold=True,
    )

    # Vector-only → rerank to get scores
    vector_chunks = run_vector_only(store, query, top_k=settings.retrieval.vector_top_k)
    vector_scored = reranker.rerank(
        query=query,
        candidates=vector_chunks[: settings.retrieval.rerank_candidates],
        top_k=settings.retrieval.rerank_top_k,
        skip_threshold=True,
    )

    # Hybrid (BM25 + vector → RRF → rerank)
    hybrid_results, bm25_ids, vector_ids = run_hybrid(
        bm25,
        store,
        reranker,
        query,
        bm25_top_k=settings.retrieval.bm25_top_k,
        vector_top_k=settings.retrieval.vector_top_k,
        rerank_candidates=settings.retrieval.rerank_candidates,
        rerank_top_k=settings.retrieval.rerank_top_k,
    )

    bm25_id_set = {sc.chunk.chunk_id for sc in bm25_scored}
    vector_id_set = {sc.chunk.chunk_id for sc in vector_scored}
    hybrid_id_set = {c.chunk_id for c, _ in hybrid_results}

    # Chunks hybrid found that vector-only missed
    hybrid_not_vector = hybrid_id_set - vector_id_set
    # Chunks hybrid found that bm25-only missed
    hybrid_not_bm25 = hybrid_id_set - bm25_id_set

    return {
        "query": query,
        "bm25_scored": bm25_scored,
        "vector_scored": vector_scored,
        "hybrid_results": hybrid_results,
        "bm25_id_set": bm25_id_set,
        "vector_id_set": vector_id_set,
        "hybrid_id_set": hybrid_id_set,
        "hybrid_not_vector": hybrid_not_vector,
        "hybrid_not_bm25": hybrid_not_bm25,
        "bm25_candidate_ids": set(bm25_ids),
        "vector_candidate_ids": set(vector_ids),
    }


def print_comparison(result: dict) -> None:
    query = result["query"]
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"QUERY: {query!r}")
    print(sep)

    # BM25-only
    print("\n  [BM25-only after rerank]")
    if result["bm25_scored"]:
        for sc in result["bm25_scored"]:
            flag = (
                " *HYBRID-MISS*"
                if sc.chunk.chunk_id not in result["hybrid_id_set"]
                else ""
            )
            print(
                f"    {_short_id(sc.chunk.chunk_id)}  score={sc.score:+.3f}"
                f"  {_preview(sc.chunk.content)}{flag}"
            )
    else:
        print("    (no results)")

    # Vector-only
    print("\n  [Vector-only after rerank]")
    if result["vector_scored"]:
        for sc in result["vector_scored"]:
            flag = (
                " *HYBRID-MISS*"
                if sc.chunk.chunk_id not in result["hybrid_id_set"]
                else ""
            )
            print(
                f"    {_short_id(sc.chunk.chunk_id)}  score={sc.score:+.3f}"
                f"  {_preview(sc.chunk.content)}{flag}"
            )
    else:
        print("    (no results)")

    # Hybrid
    print("\n  [Hybrid (BM25+Vector->RRF->rerank)]")
    if result["hybrid_results"]:
        for chunk, score in result["hybrid_results"]:
            in_bm25 = "B" if chunk.chunk_id in result["bm25_candidate_ids"] else " "
            in_vec = "V" if chunk.chunk_id in result["vector_candidate_ids"] else " "
            print(
                f"    {_short_id(chunk.chunk_id)}  score={score:+.3f}"
                f"  [{in_bm25}{in_vec}]  {_preview(chunk.content)}"
            )
    else:
        print("    (no results)")

    # Advantage summary
    if result["hybrid_not_vector"]:
        n = len(result["hybrid_not_vector"])
        print(f"\n  ** HYBRID ADVANTAGE: found {n} chunk(s) that vector-only missed **")
        # Show what those chunks look like
        for chunk, score in result["hybrid_results"]:
            if chunk.chunk_id in result["hybrid_not_vector"]:
                print(f"     -> score={score:+.3f}  {_preview(chunk.content, 70)}")

    if result["hybrid_not_bm25"]:
        n = len(result["hybrid_not_bm25"])
        print(f"\n  ** HYBRID ADVANTAGE: found {n} chunk(s) that BM25-only missed **")
        for chunk, score in result["hybrid_results"]:
            if chunk.chunk_id in result["hybrid_not_bm25"]:
                print(f"     -> score={score:+.3f}  {_preview(chunk.content, 70)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading indexes…")
    settings = get_settings()
    bm25 = BM25Index.load()
    store = ChromaStore()
    reranker = Reranker()

    print(f"BM25 chunks indexed: {bm25.chunk_count()}")
    print(f"Chroma chunks indexed: {store.count()}")
    print(f"Confidence threshold: {settings.retrieval.confidence_threshold}")
    print(f"Decline threshold: {settings.retrieval.decline_threshold}")

    hybrid_advantage_queries: list[str] = []

    for query in TEST_QUERIES:
        result = compare_query(query, bm25, store, reranker)
        print_comparison(result)
        if result["hybrid_not_vector"]:
            hybrid_advantage_queries.append(query)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Queries where hybrid found chunks that vector-only missed")
    print("=" * 80)
    if hybrid_advantage_queries:
        for q in hybrid_advantage_queries:
            print(f"  - {q!r}")
    else:
        print("  None — vector and hybrid returned the same top-k for all queries.")

    exact_term_queries = [
        "Vaswani et al.",
        "QK^T",
        "d_model / h",
        "10000",
        "Chinchilla",
    ]
    exact_with_advantage = [
        q for q in hybrid_advantage_queries if q in exact_term_queries
    ]
    print(
        f"\nExact-term queries with hybrid advantage: "
        f"{len(exact_with_advantage)}/{len(exact_term_queries)}"
    )
    for q in exact_with_advantage:
        print(f"  - {q!r}")


if __name__ == "__main__":
    main()
