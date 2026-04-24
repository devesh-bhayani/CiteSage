"""Answer generator with enforced citations and decline-to-answer.

Rules (from generation CLAUDE.md):
- Generation uses Sonnet (models.generator from config.yaml).
- Decline when no chunks pass the confidence threshold.
- All LLM calls: retry with exponential backoff (max 3 attempts, base 1 s).
- Log token usage per call.
- Prompts loaded from YAML only (never inline).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..utils.llm_factory import get_generator_llm
from ..prompts import load_prompt
from ..retrieval.retriever import RetrievalResult, ScoredChunk

load_dotenv()

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """The final output handed back to the caller (CLI, API, etc.)."""

    query: str
    answer: str
    sources: list[ScoredChunk] = field(default_factory=list)
    declined: bool = False
    token_usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """Produce a cited answer given a RetrievalResult.

    Decline-to-answer:
        If ``retrieval_result.has_relevant_chunks`` is False (i.e. no chunk
        scored above ``retrieval.confidence_threshold``), the generator
        returns the canned decline message from ``decline.yaml`` without
        making any LLM call.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.models.generator
        self._llm = get_generator_llm(max_tokens=1024)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, retrieval_result: RetrievalResult) -> GenerationResult:
        """Generate an answer for *retrieval_result*.

        Returns a ``GenerationResult`` with ``declined=True`` when there are
        no relevant chunks.
        """
        query = retrieval_result.query

        # ----- decline path -----
        if not retrieval_result.has_relevant_chunks:
            decline_prompt = load_prompt("decline")
            logger.info("generation.declined", query_preview=query[:80])
            return GenerationResult(
                query=query,
                answer=decline_prompt["message"].strip(),
                declined=True,
            )

        # ----- generation path -----
        sources_block = self._format_sources(retrieval_result.scored_chunks)
        gen_prompt = load_prompt("generate")
        system_text = gen_prompt["system"]

        user_text = f"<sources>\n{sources_block}\n</sources>\n\n" f"Question: {query}"

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ]

        response, usage = self._invoke_with_retry(messages)

        logger.info(
            "generation.complete",
            query_preview=query[:80],
            source_count=len(retrieval_result.scored_chunks),
            token_usage=usage,
        )

        return GenerationResult(
            query=query,
            answer=response,
            sources=retrieval_result.scored_chunks,
            declined=False,
            token_usage=usage,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources(scored_chunks: list[ScoredChunk]) -> str:
        """Build the numbered <sources> block injected into the user message."""
        parts: list[str] = []
        for idx, sc in enumerate(scored_chunks, start=1):
            header = f"[Source {idx}]"
            meta_parts: list[str] = [f"file: {sc.chunk.source_file}"]
            if sc.chunk.page_number:
                meta_parts.append(f"page: {sc.chunk.page_number}")
            if sc.chunk.section_heading:
                meta_parts.append(f"section: {sc.chunk.section_heading}")
            meta_line = " | ".join(meta_parts)
            parts.append(f"{header} ({meta_line})\n{sc.chunk.content}")
        return "\n\n".join(parts)

    def _invoke_with_retry(
        self,
        messages: list,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> tuple[str, dict]:
        """Call the LLM with exponential backoff.

        Returns (response_text, token_usage_dict).
        Raises the last exception if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._llm.invoke(messages)

                usage: dict = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = dict(response.usage_metadata)
                elif hasattr(response, "response_metadata"):
                    rm = response.response_metadata or {}
                    usage = rm.get("usage", {})

                return response.content, usage

            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "generation.retry",
                        attempt=attempt,
                        delay=delay,
                        error=str(exc),
                    )
                    time.sleep(delay)

        raise last_exc  # type: ignore[misc]
