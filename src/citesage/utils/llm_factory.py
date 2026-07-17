"""Provider-agnostic LLM factory for CiteSage.

Reads ``provider`` from config.yaml (default: ``"anthropic"``).  When set to
``"ollama"``, all LLM calls are routed to a local Ollama instance instead.

Usage
-----
    from citesage.utils.llm_factory import get_generator_llm, get_grader_llm

    llm = get_generator_llm()   # Sonnet-equivalent
    llm = get_grader_llm()      # Haiku-equivalent
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is unreachable or times out.

    Distinguished from generic RuntimeError so callers (retry loops, health
    checks) can react differently to infra failures vs model errors.
    """


# Read timeout for a single Ollama call. Generous (180 s) because the eval
# alternates generator (qwen3:8b) and grader (qwen3-small) every query, forcing
# a model swap; a cold reload under memory pressure can exceed 60 s and was
# aborting eval runs mid-stream on a grader ReadTimeout (~6/65 queries in).
# A genuinely dead daemon still fails fast via "connection refused" (instant,
# independent of this read timeout), so dead-daemon detection isn't slowed.
# Override with CITESAGE_OLLAMA_TIMEOUT.
_OLLAMA_TIMEOUT_S = float(os.environ.get("CITESAGE_OLLAMA_TIMEOUT", "180"))

# Sampling seed, paired with temperature=0 in OllamaLLM.invoke() to make eval
# runs reproducible. Override with CITESAGE_OLLAMA_SEED.
_OLLAMA_SEED = int(os.environ.get("CITESAGE_OLLAMA_SEED", "42"))


def _get_provider() -> str:
    """Return the configured provider name (lowercase)."""
    settings = get_settings()
    return getattr(settings, "provider", "anthropic").lower()


# ---------------------------------------------------------------------------
# Thin wrapper around the raw ``ollama`` Python client.
#
# langchain-ollama's ChatOllama silently strips <think> tags from thinking
# models (qwen3, gemma4) and returns empty content.  This wrapper calls
# ``ollama.chat()`` directly, appends ``/no_think`` to suppress reasoning
# tags, and returns a duck-type-compatible response object.
# ---------------------------------------------------------------------------


@dataclass
class _OllamaResponse:
    """Mimics the LangChain AIMessage interface used by CiteSage call sites."""

    content: str = ""
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


class OllamaLLM:
    """Lightweight wrapper around ``ollama.chat`` with LangChain-compatible
    ``.invoke(messages)`` interface."""

    def __init__(self, model: str, num_predict: int = 1024) -> None:
        self.model = model
        self.num_predict = num_predict

    @staticmethod
    def _convert_messages(messages: list) -> list[dict[str, str]]:
        """Convert LangChain message objects to ollama dict format."""
        result: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "user")
                if role == "ai":
                    role = "assistant"
                result.append({"role": role, "content": msg.content})
        return result

    def invoke(self, messages: list, **kwargs: Any) -> _OllamaResponse:
        import ollama as _ollama

        converted = self._convert_messages(messages)

        # Use a Client with an explicit timeout so connection failures surface
        # quickly. The default ollama.chat() uses a module-level client with
        # no timeout, which lets dead-daemon calls hang on socket read.
        client = _ollama.Client(timeout=_OLLAMA_TIMEOUT_S)

        # think=False asks the model to skip reasoning tags. qwen3-family
        # models ignore it and reason inline anyway, so the </think> strip
        # below is the real defense (verified: tags leak with AND without the
        # options dict — an older comment here blamed options for that, wrong).
        #
        # options pins greedy decoding so eval runs are reproducible. Without
        # it, identical eval runs varied 6-10pp because the generator, relevance
        # grader, citation judge, and eval grader all sampled independently.
        #
        # Deliberately NOT passing num_predict: the citation judge requests
        # max_tokens=16, but these models emit reasoning before the verdict, so
        # a low token cap truncates mid-reasoning and loses the YES/NO/PARTIAL.
        try:
            raw = client.chat(
                model=self.model,
                messages=converted,
                think=False,
                options={"temperature": 0, "seed": _OLLAMA_SEED},
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(
                hint in msg for hint in ("connect", "refused", "timed out", "timeout")
            ):
                raise OllamaConnectionError(
                    f"Ollama unreachable at default endpoint "
                    f"(timeout={_OLLAMA_TIMEOUT_S}s, model={self.model}): {exc}. "
                    f"Check that the Ollama daemon is running "
                    f"(https://ollama.com/download)."
                ) from exc
            raise

        content = raw.message.content or ""
        # Some models (e.g. qwen3-small) emit reasoning inline even when
        # think=False, terminated by a bare </think> tag.  Strip everything
        # up to and including the last </think> occurrence.
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        usage = {}
        if hasattr(raw, "prompt_eval_count"):
            usage["input_tokens"] = raw.prompt_eval_count or 0
        if hasattr(raw, "eval_count"):
            usage["output_tokens"] = raw.eval_count or 0

        return _OllamaResponse(
            content=content,
            usage_metadata=usage,
            response_metadata={"model": self.model},
        )


def _make_ollama(model: str, max_tokens: int) -> OllamaLLM:
    return OllamaLLM(model=model, num_predict=max_tokens)


def _make_anthropic(model: str, max_tokens: int) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=max_tokens)


_FACTORIES = {
    "anthropic": _make_anthropic,
    "ollama": _make_ollama,
}


def get_llm(model: str, max_tokens: int = 1024) -> BaseChatModel | OllamaLLM:
    """Create an LLM instance for the configured provider."""
    provider = _get_provider()
    factory = _FACTORIES.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. " f"Supported: {sorted(_FACTORIES)}"
        )
    return factory(model, max_tokens)


def get_generator_llm(max_tokens: int = 1024) -> BaseChatModel | OllamaLLM:
    """Return an LLM configured for answer generation (Sonnet-equivalent)."""
    return get_llm(get_settings().models.generator, max_tokens=max_tokens)


def get_grader_llm(max_tokens: int = 512) -> BaseChatModel | OllamaLLM:
    """Return an LLM configured for grading/routing (Haiku-equivalent)."""
    return get_llm(get_settings().models.grader, max_tokens=max_tokens)
