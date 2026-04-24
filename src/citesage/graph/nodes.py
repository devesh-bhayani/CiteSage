"""LangGraph node functions and routing logic for the CiteSage pipeline.

Node overview
-------------
SHARED:
  retrieve_node       BM25 + vector → RRF candidates (no reranking)
  rerank_node         Cross-encoder scoring, all candidates kept (no threshold)

FAST PATH  (reranker_top_score >= confidence_threshold):
  generate_fast_node  Filter above threshold → one LLM call → answer + citations

THOROUGH PATH (reranker_top_score < confidence_threshold, chunks exist):
  grade_relevance_node  Haiku batch-grades chunk relevance, filters list
  generate_thorough_node  LLM call on graded chunks
  verify_citations_node   Hybrid overlap + Haiku judge; sets confidence
  transform_query_node    Haiku rewrites question; increments retry_count

DECLINE (no chunks survive retrieval or grading):
  decline_node        Sets canned answer, confidence="low", path_taken="declined"

Routing functions (return str, used as conditional edge targets):
  route_after_rerank   "fast" | "thorough" | "decline"
  route_after_grade    "generate_thorough" | "decline"
  route_after_verify   "done" | "retry"
"""

from __future__ import annotations

import json
import re
import time

import structlog
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..utils.llm_factory import get_generator_llm, get_grader_llm
from ..generation.citation_verifier import CitationVerifier
from ..ingestion.models import Chunk
from ..prompts import load_prompt
from ..retrieval._types import ScoredChunk
from ..retrieval.reranker import Reranker
from ..retrieval.retriever import Retriever
from .state import RAGState

load_dotenv()

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy LLM helpers
# ---------------------------------------------------------------------------


def _haiku() -> BaseChatModel:
    return get_grader_llm(max_tokens=512)


def _sonnet() -> BaseChatModel:
    return get_generator_llm(max_tokens=1024)


def _llm_invoke_with_retry(
    llm: BaseChatModel,
    messages: list,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> tuple[str, dict]:
    """Invoke *llm* with exponential backoff.  Returns (text, usage_dict)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(messages)
            usage: dict = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = dict(response.usage_metadata)
            elif hasattr(response, "response_metadata"):
                usage = (response.response_metadata or {}).get("usage", {})
            return response.content, usage
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "llm.retry", attempt=attempt, delay=delay, error=str(exc)
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _merge_usage(base: dict, new: dict) -> dict:
    """Accumulate flat token counts across multiple LLM calls."""
    merged = dict(base)
    for k, v in new.items():
        if isinstance(v, (int, float)):
            merged[k] = merged.get(k, 0) + v
        else:
            merged.setdefault(k, v)
    return merged


def _merge_model_usage(base: dict, model_name: str, usage: dict) -> dict:
    """Accumulate per-model token counts and call count in *base*.

    ``base`` has shape ``{model_id: {input_tokens, output_tokens, calls}}``.
    Returns a new dict; does not mutate *base*.
    """
    merged = {k: dict(v) for k, v in base.items()}
    entry = merged.setdefault(
        model_name, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    )
    entry["input_tokens"] += usage.get("input_tokens", 0)
    entry["output_tokens"] += usage.get("output_tokens", 0)
    entry["calls"] += 1
    return merged


# ---------------------------------------------------------------------------
# Citation helpers (shared between generate and verify nodes)
# ---------------------------------------------------------------------------


def _format_sources(scored_chunks: list[ScoredChunk]) -> str:
    parts: list[str] = []
    for idx, sc in enumerate(scored_chunks, start=1):
        header = f"[Source {idx}]"
        meta: list[str] = [f"file: {sc.chunk.source_file}"]
        if sc.chunk.page_number:
            meta.append(f"page: {sc.chunk.page_number}")
        if sc.chunk.section_heading:
            meta.append(f"section: {sc.chunk.section_heading}")
        parts.append(f"{header} ({' | '.join(meta)})\n{sc.chunk.content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    """BM25 + vector search → RRF fusion → raw candidates stored in state."""
    retriever = Retriever()
    candidates, log_info = retriever.retrieve_candidates(state["question"])
    logger.info("graph.retrieve", **log_info, candidates=len(candidates))
    return {"retrieved_chunks": candidates}


# ---------------------------------------------------------------------------
# Node: rerank
# ---------------------------------------------------------------------------


def rerank_node(state: RAGState) -> dict:
    """Cross-encoder scores all candidates; threshold NOT applied here.

    The top score is stored so that ``route_after_rerank`` can decide which
    path to take without re-scoring.
    """
    candidates: list[Chunk] = state["retrieved_chunks"]
    if not candidates:
        return {"reranked_chunks": [], "reranker_top_score": 0.0}

    reranker = Reranker()
    scored = reranker.rerank(
        query=state["question"],
        candidates=candidates,
        skip_threshold=True,  # keep all; routing decides what to do
    )
    top_score = scored[0].score if scored else 0.0
    logger.info(
        "graph.rerank", candidates_in=len(candidates), top_score=round(top_score, 3)
    )
    return {"reranked_chunks": scored, "reranker_top_score": top_score}


# ---------------------------------------------------------------------------
# Routing: after rerank
# ---------------------------------------------------------------------------


def route_after_rerank(state: RAGState) -> str:
    """Choose FAST, THOROUGH, or DECLINE based on reranker top score.

    Decision boundaries (all configurable in config.yaml):
      top_score < decline_threshold   → "decline"  (clearly irrelevant, no LLM)
      top_score >= confidence_threshold → "fast"    (high-confidence, 1 LLM call)
      otherwise                         → "thorough" (uncertain, 3-4 LLM calls)
    """
    if not state["reranked_chunks"]:
        return "decline"
    settings = get_settings()
    top = state["reranker_top_score"]
    if top < settings.retrieval.decline_threshold:
        return "decline"
    if top >= settings.retrieval.confidence_threshold:
        return "fast"
    return "thorough"


# ---------------------------------------------------------------------------
# Node: decline
# ---------------------------------------------------------------------------


def decline_node(state: RAGState) -> dict:
    """Set the canned decline answer (no LLM call)."""
    msg = load_prompt("decline")["message"].strip()
    logger.info("graph.decline", question_preview=state["question"][:80])
    return {
        "answer": msg,
        "citations": [],
        "confidence": "low",
        "path_taken": "declined",
    }


# ---------------------------------------------------------------------------
# Node: generate_fast  (FAST PATH)
# ---------------------------------------------------------------------------


def generate_fast_node(state: RAGState) -> dict:
    """FAST PATH: filter chunks above confidence_threshold, one LLM call."""
    settings = get_settings()
    threshold = settings.retrieval.confidence_threshold
    above: list[ScoredChunk] = [
        sc for sc in state["reranked_chunks"] if sc.score >= threshold
    ]

    if not above:
        # Threshold filtering removed everything — fall back to decline.
        msg = load_prompt("decline")["message"].strip()
        return {
            "answer": msg,
            "citations": [],
            "confidence": "low",
            "path_taken": "declined",
        }

    sources_block = _format_sources(above)
    gen_prompt = load_prompt("generate")
    messages = [
        SystemMessage(content=gen_prompt["system"]),
        HumanMessage(
            content=f"<sources>\n{sources_block}\n</sources>\n\nQuestion: {state['question']}"
        ),
    ]

    model_name = settings.models.generator
    llm = _sonnet()
    text, usage = _llm_invoke_with_retry(llm, messages)
    logger.info("graph.generate_fast", sources=len(above), token_usage=usage)

    # Verify citations even on the fast path — skipping verification was
    # the primary cause of near-zero citation_precision on fast-path queries.
    verifier = CitationVerifier()
    vresult = verifier.verify(text, above)
    logger.info(
        "graph.generate_fast.verify",
        total=vresult.total_cited,
        supported=vresult.supported_count,
        confidence=vresult.confidence,
    )

    accumulated_usage = _merge_usage(state.get("token_usage", {}), usage)
    accumulated_model = _merge_model_usage(
        state.get("model_usage", {}), model_name, usage
    )
    if vresult.token_usage:
        grader_name = settings.models.grader
        accumulated_usage = _merge_usage(accumulated_usage, vresult.token_usage)
        accumulated_model = _merge_model_usage(
            accumulated_model, grader_name, vresult.token_usage
        )

    return {
        "answer": text,
        "citations": above,
        "confidence": vresult.confidence,
        "path_taken": "fast",
        "token_usage": accumulated_usage,
        "model_usage": accumulated_model,
    }


# ---------------------------------------------------------------------------
# Node: grade_relevance  (THOROUGH PATH)
# ---------------------------------------------------------------------------


_GRADE_ARRAY_RE = re.compile(r"\[[^\[\]]*\]")


def _parse_grade_indices(text: str, *, total: int) -> list[int]:
    """Parse the grader's JSON array response into 1-based chunk indices.

    Tries strict JSON first, then falls back to extracting the first JSON-array
    substring (handles models that wrap responses in prose or markdown fences).
    If both fail, returns []: route_after_grade will then decline the query
    rather than answer from an unfiltered set — this is the safe choice for
    unanswerable questions where the grader is the last defense.
    """
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    match = _GRADE_ARRAY_RE.search(text)
    if match and match.group(0) not in candidates:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(raw, list):
            return [
                int(x)
                for x in raw
                if isinstance(x, (int, float)) and 1 <= int(x) <= total
            ]

    logger.warning(
        "graph.grade_relevance.parse_failed",
        raw_response=text[:200],
        total_chunks=total,
    )
    return []


def grade_relevance_node(state: RAGState) -> dict:
    """Haiku batch-grades chunk relevance; filters reranked_chunks in-place."""
    chunks: list[ScoredChunk] = state["reranked_chunks"]
    if not chunks:
        return {}

    numbered_chunks = "\n\n".join(
        f"[{i}] {sc.chunk.content[:400]}" for i, sc in enumerate(chunks, start=1)
    )
    grade_prompt = load_prompt("grade")
    messages = [
        SystemMessage(content=grade_prompt["system"]),
        HumanMessage(
            content=f"Question: {state['question']}\n\nChunks:\n{numbered_chunks}"
        ),
    ]

    model_name = get_settings().models.grader
    llm = _haiku()
    text, usage = _llm_invoke_with_retry(llm, messages)
    logger.info("graph.grade_relevance", raw_response=text[:120])

    relevant_indices = _parse_grade_indices(text, total=len(chunks))

    # Convert 1-based indices to filtered list.
    kept = [chunks[i - 1] for i in relevant_indices if 1 <= i <= len(chunks)]
    logger.info("graph.grade_relevance", before=len(chunks), after=len(kept))

    return {
        "reranked_chunks": kept,
        "token_usage": _merge_usage(state.get("token_usage", {}), usage),
        "model_usage": _merge_model_usage(
            state.get("model_usage", {}), model_name, usage
        ),
    }


# ---------------------------------------------------------------------------
# Routing: after grade_relevance
# ---------------------------------------------------------------------------


def route_after_grade(state: RAGState) -> str:
    """If grading removed all chunks, decline; otherwise generate."""
    return "decline" if not state["reranked_chunks"] else "generate_thorough"


# ---------------------------------------------------------------------------
# Node: generate_thorough  (THOROUGH PATH)
# ---------------------------------------------------------------------------


def generate_thorough_node(state: RAGState) -> dict:
    """THOROUGH PATH: generate from graded chunks (no score threshold applied)."""
    chunks: list[ScoredChunk] = state["reranked_chunks"]

    sources_block = _format_sources(chunks)
    gen_prompt = load_prompt("generate")
    messages = [
        SystemMessage(content=gen_prompt["system"]),
        HumanMessage(
            content=f"<sources>\n{sources_block}\n</sources>\n\nQuestion: {state['question']}"
        ),
    ]

    model_name = get_settings().models.generator
    llm = _sonnet()
    text, usage = _llm_invoke_with_retry(llm, messages)
    logger.info("graph.generate_thorough", sources=len(chunks), token_usage=usage)

    return {
        "answer": text,
        "citations": chunks,
        "path_taken": "thorough",
        "token_usage": _merge_usage(state.get("token_usage", {}), usage),
        "model_usage": _merge_model_usage(
            state.get("model_usage", {}), model_name, usage
        ),
    }


# ---------------------------------------------------------------------------
# Node: verify_citations  (THOROUGH PATH)
# ---------------------------------------------------------------------------


def verify_citations_node(state: RAGState) -> dict:
    """Hybrid citation check via CitationVerifier (token overlap + Haiku judge).

    Delegates to :class:`~citesage.generation.CitationVerifier` which applies
    a 0.3 overlap threshold before escalating to the LLM judge.

    confidence mapping:
      "high"    → all cited sources supported
      "partial" → some PARTIAL verdicts, but ≤50% unsupported
      "low"     → >50% unsupported (triggers retry in route_after_verify)
    """
    answer = state.get("answer", "")
    sources: list[ScoredChunk] = state.get("citations", [])

    verifier = CitationVerifier()
    result = verifier.verify(answer, sources)

    logger.info(
        "graph.verify_citations",
        total=result.total_cited,
        supported=result.supported_count,
        partial=result.partial_count,
        unsupported=result.unsupported_count,
        confidence=result.confidence,
    )

    updates: dict = {"confidence": result.confidence}
    if result.llm_calls > 0:
        model_name = get_settings().models.grader
        prev_model_usage = state.get("model_usage", {})
        new_model_usage = {k: dict(v) for k, v in prev_model_usage.items()}
        entry = new_model_usage.setdefault(
            model_name, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        entry["input_tokens"] += result.token_usage.get("input_tokens", 0)
        entry["output_tokens"] += result.token_usage.get("output_tokens", 0)
        entry["calls"] += result.llm_calls  # one entry per judge call
        updates["token_usage"] = _merge_usage(
            state.get("token_usage", {}), result.token_usage
        )
        updates["model_usage"] = new_model_usage
    return updates


# ---------------------------------------------------------------------------
# Routing: after verify_citations
# ---------------------------------------------------------------------------


def route_after_verify(state: RAGState) -> str:
    """Retry (rewrite query) once if confidence is low; otherwise done."""
    if state.get("confidence") == "low" and state.get("retry_count", 0) < 1:
        return "retry"
    return "done"


# ---------------------------------------------------------------------------
# Node: transform_query  (THOROUGH PATH retry)
# ---------------------------------------------------------------------------


def transform_query_node(state: RAGState) -> dict:
    """Haiku rewrites the question; increments retry_count."""
    transform_prompt = load_prompt("transform_query")
    messages = [
        SystemMessage(content=transform_prompt["system"]),
        HumanMessage(content=state["question"]),
    ]

    model_name = get_settings().models.grader
    llm = _haiku()
    new_question, usage = _llm_invoke_with_retry(llm, messages)
    new_question = new_question.strip()

    logger.info(
        "graph.transform_query",
        original=state["question"][:80],
        rewritten=new_question[:80],
    )

    return {
        "question": new_question,
        "retry_count": state.get("retry_count", 0) + 1,
        # Clear stale retrieval state for the new question.
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "reranker_top_score": 0.0,
        "answer": "",
        "citations": [],
        "confidence": "",
        "token_usage": _merge_usage(state.get("token_usage", {}), usage),
        "model_usage": _merge_model_usage(
            state.get("model_usage", {}), model_name, usage
        ),
    }
