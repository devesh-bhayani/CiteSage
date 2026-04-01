"""CiteSage LangGraph pipeline.

Two paths, chosen automatically after cross-encoder reranking:

  FAST PATH   (top reranker score >= confidence_threshold):
    retrieve → rerank → generate_fast → END
    One Sonnet call.  p50 latency ~2 s.

  THOROUGH PATH (top reranker score < confidence_threshold):
    retrieve → rerank → grade_relevance → generate_thorough
             → verify_citations → END
    With one optional retry via transform_query → retrieve.
    3–4 LLM calls.  p50 latency ~5 s.

  DECLINE (no chunks survive retrieval or relevance grading):
    → decline → END
    Zero LLM calls.

Entry point
-----------
    from citesage.graph.pipeline import run_pipeline
    result = run_pipeline("What is self-attention?")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from langgraph.graph import END, StateGraph

from ..config import get_settings
from ..retrieval._types import ScoredChunk
from ..utils.cost_tracker import CostTracker, QueryCost
from .nodes import (
    decline_node,
    generate_fast_node,
    generate_thorough_node,
    grade_relevance_node,
    rerank_node,
    retrieve_node,
    route_after_grade,
    route_after_rerank,
    route_after_verify,
    transform_query_node,
    verify_citations_node,
)
from .state import RAGState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """What the CLI and API receive back from the graph."""

    answer: str
    citations: list[ScoredChunk] = field(default_factory=list)
    confidence: str = "high"  # "high" | "low"
    path_taken: str = ""  # "fast" | "thorough" | "declined"
    declined: bool = False
    token_usage: dict = field(default_factory=dict)
    query_cost: QueryCost | None = None


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    g = StateGraph(RAGState)

    # Nodes
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("decline", decline_node)
    g.add_node("generate_fast", generate_fast_node)
    g.add_node("grade_relevance", grade_relevance_node)
    g.add_node("generate_thorough", generate_thorough_node)
    g.add_node("verify_citations", verify_citations_node)
    g.add_node("transform_query", transform_query_node)

    # Entry
    g.set_entry_point("retrieve")

    # Fixed edges
    g.add_edge("retrieve", "rerank")
    g.add_edge("decline", END)
    g.add_edge("generate_fast", END)
    g.add_edge("generate_thorough", "verify_citations")

    # Routing after rerank: fast | thorough | decline
    g.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "fast": "generate_fast",
            "thorough": "grade_relevance",
            "decline": "decline",
        },
    )

    # Routing after grade: generate_thorough | decline
    g.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {
            "generate_thorough": "generate_thorough",
            "decline": "decline",
        },
    )

    # Routing after verify: done | retry (→ transform_query → retrieve loop)
    g.add_conditional_edges(
        "verify_citations",
        route_after_verify,
        {
            "done": END,
            "retry": "transform_query",
        },
    )

    # Retry loop: transform_query rewrites the question, then re-retrieves.
    g.add_edge("transform_query", "retrieve")

    return g


# Compile once at import time; reuse across queries.
_app = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pipeline(question: str) -> PipelineResult:
    """Run the full CiteSage pipeline for *question*.

    Creates an initial RAGState, invokes the compiled graph, and returns a
    PipelineResult with the answer, citations, path taken, and token usage.
    """
    initial_state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "reranker_top_score": 0.0,
        "answer": "",
        "citations": [],
        "confidence": "",
        "path_taken": "",
        "retry_count": 0,
        "error": None,
        "token_usage": {},
        "model_usage": {},
    }

    final_state: RAGState = _app.invoke(initial_state)

    path = final_state.get("path_taken", "")
    model_usage = final_state.get("model_usage", {})

    # Compute per-query cost from per-model token counts.
    settings = get_settings()
    tracker = CostTracker(settings.pricing.as_dict())
    query_cost = tracker.compute(model_usage)

    logger.info(
        "pipeline.complete",
        question_preview=question[:80],
        path_taken=path,
        confidence=final_state.get("confidence", ""),
        retry_count=final_state.get("retry_count", 0),
        cost_usd=round(query_cost.total_cost_usd, 6),
    )
    if query_cost.model_breakdown:
        logger.info("pipeline.cost", summary=query_cost.format_summary())

    return PipelineResult(
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", ""),
        path_taken=path,
        declined=(path == "declined"),
        token_usage=final_state.get("token_usage", {}),
        query_cost=query_cost,
    )
