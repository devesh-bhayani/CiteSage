"""Hybrid citation verifier for CiteSage generation layer.

Three-step process
------------------
Step 1 — Deterministic token-overlap check (no LLM):
    Extract key tokens from the claim context around each [Source N] citation
    and compute overlap against the cited chunk.  If overlap >= WEAK_THRESHOLD
    the citation is marked supported without an LLM call.

Step 2 — LLM judge for WEAK citations only:
    Citations with overlap < WEAK_THRESHOLD are sent individually to Haiku,
    which returns YES / NO / PARTIAL.

Step 3 — Decision:
    * supported_count  = YES + deterministic-pass
    * partial_count    = PARTIAL
    * unsupported_count = NO
    confidence:
      "high"    — all cited sources supported (no PARTIAL, no NO)
      "partial"  — at least one PARTIAL, but ≤50% unsupported
      "low"     — >50% of cited sources are unsupported → caller should decline
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import structlog
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.llm_factory import get_grader_llm
from ..prompts import load_prompt
from ..retrieval._types import ScoredChunk

load_dotenv()

logger = structlog.get_logger(__name__)


def _merge_usage(base: dict, new: dict) -> dict:
    merged = dict(base)
    for k, v in new.items():
        if isinstance(v, (int, float)):
            merged[k] = merged.get(k, 0) + v
        else:
            merged.setdefault(k, v)
    return merged


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of hybrid citation verification."""

    confidence: str  # "high" | "partial" | "low"
    total_cited: int = 0  # number of [Source N] refs found
    supported_count: int = 0  # deterministic pass or LLM YES
    partial_count: int = 0  # LLM PARTIAL
    unsupported_count: int = 0  # LLM NO or fabricated index
    weak_indices: list[int] = field(default_factory=list)  # sent to LLM
    unsupported_indices: list[int] = field(default_factory=list)
    # Accumulated token usage from all internal LLM judge calls.
    token_usage: dict = field(default_factory=dict)
    llm_calls: int = 0  # number of Haiku calls made


# ---------------------------------------------------------------------------
# CitationVerifier
# ---------------------------------------------------------------------------


class CitationVerifier:
    """Verify that citations in a generated answer are grounded in the sources.

    Parameters
    ----------
    weak_threshold:
        Token-overlap ratio below which a citation is considered *weak* and
        escalated to the LLM judge.  Default 0.3 (30 % token overlap).
    """

    WEAK_THRESHOLD: float = 0.55

    def __init__(self, weak_threshold: float | None = None) -> None:
        self._threshold = (
            weak_threshold if weak_threshold is not None else self.WEAK_THRESHOLD
        )
        self._llm = get_grader_llm(max_tokens=16)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, answer: str, sources: list[ScoredChunk]) -> VerificationResult:
        """Run the full three-step verification and return a :class:`VerificationResult`.

        Parameters
        ----------
        answer:
            The LLM-generated answer text, expected to contain ``[Source N]``
            reference markers.
        sources:
            Ordered list of ``ScoredChunk`` objects corresponding to the
            ``[Source 1]`` … ``[Source N]`` numbers in *answer*.
        """
        cited_indices = self._extract_cited_indices(answer)
        if not cited_indices or not sources:
            return VerificationResult(confidence="high")

        verify_prompt = load_prompt("verify_citation")
        result = VerificationResult(confidence="high", total_cited=len(cited_indices))
        weak_indices: list[int] = []

        for idx in sorted(cited_indices):
            if idx < 1 or idx > len(sources):
                # Fabricated source number — count as unsupported immediately.
                result.unsupported_count += 1
                result.unsupported_indices.append(idx)
                continue

            source_content = sources[idx - 1].chunk.content
            claim_context = self._extract_claim_context(answer, idx)
            overlap = self._token_overlap(claim_context, source_content)

            if overlap >= self._threshold:
                result.supported_count += 1
            else:
                weak_indices.append(idx)

        result.weak_indices = weak_indices

        # Step 2: LLM judge for weak citations.
        accumulated_usage: dict = {}
        for idx in weak_indices:
            source_content = sources[idx - 1].chunk.content
            claim_context = self._extract_claim_context(answer, idx)
            verdict, usage = self._llm_judge(
                verify_prompt, claim_context, source_content
            )
            accumulated_usage = _merge_usage(accumulated_usage, usage)
            result.llm_calls += 1

            if verdict == "YES":
                result.supported_count += 1
            elif verdict == "PARTIAL":
                result.partial_count += 1
            else:  # NO or error
                result.unsupported_count += 1
                result.unsupported_indices.append(idx)

        result.token_usage = accumulated_usage

        # Step 3: Determine confidence.
        total = result.total_cited
        unsupported = result.unsupported_count
        partial = result.partial_count

        if total == 0 or unsupported == 0 and partial == 0:
            result.confidence = "high"
        elif total > 0 and unsupported / total > 0.5:
            result.confidence = "low"
        elif partial > 0 or unsupported > 0:
            result.confidence = "partial"
        else:
            result.confidence = "high"

        logger.info(
            "citation_verifier.done",
            total=total,
            supported=result.supported_count,
            partial=partial,
            unsupported=unsupported,
            confidence=result.confidence,
            weak_sent_to_llm=len(weak_indices),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cited_indices(answer: str) -> set[int]:
        """Return all 1-based source numbers referenced in *answer*."""
        return {int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer)}

    @staticmethod
    def _extract_claim_context(answer: str, idx: int) -> str:
        """Return the sentence(s) in *answer* that cite ``[Source idx]``."""
        pattern = re.compile(
            rf"([^.!?]*\[Source\s+{idx}\][^.!?]*[.!?]?)", re.IGNORECASE
        )
        context = " ".join(m.group(0) for m in pattern.finditer(answer))
        return context.strip() or answer[:300]

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Fraction of *text_a*'s significant tokens that appear in *text_b*.

        Significant tokens are words of 4+ characters (case-insensitive).
        Returns 0.0 when *text_a* contains no significant tokens.
        """

        def _tokens(t: str) -> set[str]:
            return set(re.findall(r"\b\w{4,}\b", t.lower()))

        a_tokens = _tokens(text_a)
        if not a_tokens:
            return 0.0
        return len(a_tokens & _tokens(text_b)) / len(a_tokens)

    def _llm_judge(
        self,
        verify_prompt: dict,
        claim: str,
        source: str,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> tuple[str, dict]:
        """Call Haiku to judge whether *source* supports *claim*.

        Returns ``("YES"|"NO"|"PARTIAL", usage_dict)``.
        Defaults to ``"YES"`` on error to avoid false declines.
        """
        msg_text = f"Claim: {claim[:400]}\n\nSource: {source[:600]}"
        messages = [
            SystemMessage(content=verify_prompt["system"]),
            HumanMessage(content=msg_text),
        ]
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._llm.invoke(messages)
                usage: dict = {}
                um = getattr(response, "usage_metadata", None)
                if isinstance(um, dict):
                    usage = um
                elif not um:
                    rm = getattr(response, "response_metadata", None)
                    if isinstance(rm, dict):
                        usage = rm.get("usage", {})
                raw = response.content.strip().upper()
                if raw in ("YES", "NO", "PARTIAL"):
                    return raw, usage
                for keyword in ("PARTIAL", "YES", "NO"):
                    if keyword in raw:
                        return keyword, usage
                logger.warning(
                    "citation_verifier.unexpected_response",
                    raw=raw[:40],
                    defaulting_to="NO",
                )
                return "NO", usage
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "citation_verifier.retry",
                        attempt=attempt,
                        delay=delay,
                        error=str(exc),
                    )
                    time.sleep(delay)
        logger.error("citation_verifier.llm_failed", error=str(last_exc))
        return "PARTIAL", {}  # conservative: flag for review, don't silently approve
