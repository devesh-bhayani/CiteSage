"""Retrieval recall diagnostic + config sweep against the golden dataset.

Measures, with no LLM calls, how often the expected source chunk survives each
stage of retrieval for the answerable golden questions:

    BM25 top-k / vector top-k  ->  RRF pool (rerank_candidates)  ->  final top-k

and classifies every miss so tuning targets the stage that is actually losing
the chunk:

    both-miss     expected chunk in neither BM25 nor vector top-k
    rrf-dilution  retrieved, but pushed out of the candidate pool by fusion
    rerank-drop   in the pool, but the cross-encoder ranked it below top-k

BM25/vector rankings and cross-encoder scores are computed once per question
over the whole corpus, so ``--sweep`` re-slices them for every config instead
of re-running retrieval (the sweep is then effectively instant).

Run from the project root::

    uv run python scripts/retrieval_recall.py
    uv run python scripts/retrieval_recall.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, "src")

from citesage.config import get_settings
from citesage.ingestion.storage import BM25Index, ChromaStore
from citesage.retrieval.reranker import Reranker
from citesage.retrieval.rrf import rrf_fuse

DATASET = "tests/eval/golden_dataset.json"


def load_answerable(path: str = DATASET) -> list[dict[str, Any]]:
    """Return the golden questions that have at least one expected chunk.

    Args:
        path: Path to the golden dataset JSON.

    Returns:
        The answerable question dicts (unanswerable ones have no
        ``expected_source_chunks`` and have no retrieval target to measure).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [q for q in data if q.get("expected_source_chunks")]


def precompute(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every corpus chunk against every question, once.

    Args:
        questions: Answerable golden questions.

    Returns:
        Per-question dicts holding the full BM25 and vector rankings plus a
        ``chunk_id -> cross-encoder score`` map covering the whole corpus.
    """
    store = ChromaStore()
    bm25 = BM25Index.load()
    model = Reranker()._model  # noqa: SLF001 - diagnostic script, not library code
    n_corpus = bm25.chunk_count()

    out: list[dict[str, Any]] = []
    for q in questions:
        query = q["question"]
        bm25_raw = bm25.search(query, top_k=n_corpus)
        vector_raw = store.query(query_text=query, top_k=n_corpus)

        by_id = {c.chunk_id: c for c, _ in bm25_raw}
        for c, _ in vector_raw:
            by_id.setdefault(c.chunk_id, c)

        ids = list(by_id)
        scores = model.predict(
            [(query, by_id[c].content) for c in ids], show_progress_bar=False
        )
        out.append(
            {
                "id": q["id"],
                "category": q["category"],
                "expected": set(q["expected_source_chunks"]),
                "bm25_list": [(c.chunk_id, c) for c, _ in bm25_raw],
                "vector_list": [(c.chunk_id, c) for c, _ in vector_raw],
                "ce_score": {c: float(s) for c, s in zip(ids, scores)},
            }
        )
    return out


def evaluate(
    precomp: list[dict[str, Any]],
    bm25_k: int,
    vec_k: int,
    rerank_cand: int,
    rerank_k: int,
    rrf_k: int,
) -> tuple[dict[str, int], dict[str, dict[str, int]], list[tuple[str, str, str]]]:
    """Evaluate one retrieval config over the precomputed rankings.

    A question counts as a hit at a stage when any of its expected chunks is
    present in that stage's output.

    Args:
        precomp: Output of :func:`precompute`.
        bm25_k: BM25 top-k.
        vec_k: Vector top-k.
        rerank_cand: Candidates fed to the cross-encoder.
        rerank_k: Final results handed to generation.
        rrf_k: RRF smoothing constant.

    Returns:
        Tuple of (totals, per-category counts, misses) where each miss is
        ``(question_id, category, classification)``.
    """
    stats = {"bm25": 0, "vector": 0, "pool": 0, "final": 0, "total": len(precomp)}
    per_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pool": 0, "final": 0, "total": 0}
    )
    misses: list[tuple[str, str, str]] = []

    for p in precomp:
        expected: set[str] = p["expected"]
        cat = p["category"]
        per_cat[cat]["total"] += 1

        bm25_list = p["bm25_list"][:bm25_k]
        vec_list = p["vector_list"][:vec_k]
        in_bm25 = bool(expected & {cid for cid, _ in bm25_list})
        in_vec = bool(expected & {cid for cid, _ in vec_list})
        stats["bm25"] += in_bm25
        stats["vector"] += in_vec

        fused = rrf_fuse([bm25_list, vec_list], k=rrf_k)
        pool = [cid for cid, _, _ in fused[:rerank_cand]]
        in_pool = bool(expected & set(pool))
        stats["pool"] += in_pool
        per_cat[cat]["pool"] += in_pool

        ranked = sorted(pool, key=lambda c: p["ce_score"][c], reverse=True)[:rerank_k]
        in_final = bool(expected & set(ranked))
        stats["final"] += in_final
        per_cat[cat]["final"] += in_final

        if not in_final:
            if not in_bm25 and not in_vec:
                cls = "both-miss"
            elif not in_pool:
                cls = "rrf-dilution"
            else:
                cls = "rerank-drop"
            misses.append((p["id"], cat, cls))

    return stats, dict(per_cat), misses


def _report(precomp: list[dict[str, Any]], cfg: tuple[int, int, int, int, int]) -> None:
    """Print the stage-by-stage recall breakdown and miss classification."""
    stats, per_cat, misses = evaluate(precomp, *cfg)
    t = stats["total"]
    print(
        f"config: bm25={cfg[0]} vector={cfg[1]} candidates={cfg[2]} "
        f"top_k={cfg[3]} rrf_k={cfg[4]}\n"
    )
    for stage, label in (
        ("bm25", f"BM25 top-{cfg[0]}"),
        ("vector", f"vector top-{cfg[1]}"),
        ("pool", f"RRF pool top-{cfg[2]}"),
        ("final", f"final top-{cfg[3]}"),
    ):
        print(f"  {label:22s} {stats[stage]:2d}/{t}  {stats[stage] / t:4.0%}")

    print("\n  per category (pool / final / total):")
    for cat, d in sorted(per_cat.items()):
        print(f"    {cat:16s} {d['pool']:2d} / {d['final']:2d} / {d['total']:2d}")

    counts: dict[str, int] = defaultdict(int)
    for _, _, cls in misses:
        counts[cls] += 1
    print(
        f"\n  misses ({len(misses)}): "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    for mid, cat, cls in misses:
        print(f"    {mid:24s} {cat:16s} {cls}")


def _sweep(precomp: list[dict[str, Any]], rrf_k: int) -> None:
    """Print final/pool recall for a grid of retrieval configs."""
    print(f"{'bm25/vec':>8s} {'cand':>5s} {'top_k':>6s} | {'pool':>5s} {'final':>6s}")
    print("-" * 40)
    rows = []
    for kk in (20, 25, 30):
        for cand in (15, 20, 25, 30):
            for topk in (5, 7):
                stats, _, _ = evaluate(precomp, kk, kk, cand, topk, rrf_k)
                t = stats["total"]
                rows.append((stats["final"] / t, stats["pool"] / t, (kk, cand, topk)))
                print(
                    f"{kk:>8d} {cand:>5d} {topk:>6d} | "
                    f"{stats['pool'] / t:5.0%} {stats['final'] / t:6.0%}"
                )
    rows.sort(key=lambda r: (-r[0], -r[1], r[2][2], r[2][1]))
    print("\nbest by final recall (tie-break: smaller top_k, then smaller pool):")
    for final, pool, (kk, cand, topk) in rows[:5]:
        print(
            f"  bm25/vec={kk} cand={cand} top_k={topk} -> final {final:.0%} pool {pool:.0%}"
        )


def main() -> None:
    """Entry point: report current config, optionally sweep alternatives."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="grid-sweep retrieval knobs")
    args = ap.parse_args()

    r = get_settings().retrieval
    questions = load_answerable()
    print(f"answerable questions: {len(questions)}  (scoring corpus, ~30s)\n")
    precomp = precompute(questions)

    _report(
        precomp,
        (r.bm25_top_k, r.vector_top_k, r.rerank_candidates, r.rerank_top_k, r.rrf_k),
    )
    if args.sweep:
        print()
        _sweep(precomp, r.rrf_k)


if __name__ == "__main__":
    main()
