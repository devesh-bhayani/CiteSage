"""CiteSage CLI — ingest documents and ask questions with cited answers.

Usage:
    # Ingest documents
    python -m citesage.cli --ingest path/to/docs/

    # Ask a question
    python -m citesage.cli "What is reinforcement learning?"

    # Both: ingest first, then ask
    python -m citesage.cli --ingest data/documents/ "What is X?"
"""

from __future__ import annotations

import argparse
import sys

import structlog

logger = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citesage",
        description="Document QA with verified citations.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer from ingested documents.",
    )
    parser.add_argument(
        "--ingest",
        metavar="PATH",
        help="File or directory to ingest before querying.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10).",
    )
    return parser


def _run_ingest(path: str) -> None:
    """Ingest a file or directory."""
    from pathlib import Path

    from .ingestion.pipeline import IngestPipeline

    target = Path(path)
    pipeline = IngestPipeline()

    if target.is_dir():
        chunks = pipeline.ingest_directory(target)
    elif target.is_file():
        chunks = pipeline.ingest_file(target)
    else:
        print(f"Error: '{path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {len(chunks)} chunk(s) from {path}")


def _run_query(question: str) -> None:
    """Run the LangGraph pipeline and print the answer with citations."""
    from .graph.pipeline import run_pipeline

    result = run_pipeline(question)

    # -- Print answer --
    print()
    print(result.answer)

    # -- Print path / confidence metadata --
    meta_parts = [f"path={result.path_taken}"]
    if result.confidence:
        meta_parts.append(f"confidence={result.confidence}")
    print(f"\n[{', '.join(meta_parts)}]")

    # -- Print source list --
    if result.citations:
        print("\n--- Sources ---")
        for idx, sc in enumerate(result.citations, start=1):
            meta = f"{sc.chunk.source_file}"
            if sc.chunk.page_number:
                meta += f" (p.{sc.chunk.page_number})"
            print(f"  [{idx}] {meta}  (score: {sc.score:.3f})")

    # -- Print token usage and cost --
    if result.token_usage:
        usage = result.token_usage
        input_tok = usage.get("input_tokens", "?")
        output_tok = usage.get("output_tokens", "?")
        print(f"\nTokens: {input_tok} in / {output_tok} out")

    if result.query_cost is not None:
        print(result.query_cost.format_summary())


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.ingest is None and args.question is None:
        parser.print_help()
        sys.exit(0)

    if args.ingest:
        _run_ingest(args.ingest)

    if args.question:
        _run_query(args.question)


if __name__ == "__main__":
    main()
