"""CiteSage configuration loader.

Loads config.yaml and exposes typed settings via get_settings().
Resolution order: CITESAGE_CONFIG env var → explicit path → walk up from cwd.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChunkingStrategyConfig(BaseModel):
    """Chunking parameters for one document type."""

    size: int
    overlap: int
    strategy: str = "recursive"


class ChunkingConfig(BaseModel):
    """Root chunking config.  Extra keys are per-doc-type overrides."""

    model_config = {"extra": "allow"}

    default: ChunkingStrategyConfig

    def for_doc_type(self, doc_type: str) -> ChunkingStrategyConfig:
        """Return chunking config for *doc_type*, falling back to default."""
        extra = self.model_extra or {}
        if doc_type in extra:
            return ChunkingStrategyConfig(**extra[doc_type])
        return self.default


class ModelsConfig(BaseModel):
    generator: str
    grader: str
    embedder: str
    reranker: str


class RetrievalConfig(BaseModel):
    vector_top_k: int
    bm25_top_k: int
    # Number of RRF-fused candidates fed to the cross-encoder.
    rerank_candidates: int = 15
    # Final number of results returned after reranking.
    rerank_top_k: int
    rrf_k: int
    # Floor for raw cosine similarity from the vector store (pre-rerank).
    vector_score_threshold: float = 0.3
    # Floor for the cross-encoder reranker score (Phase 2+).
    confidence_threshold: float
    # Hard-decline threshold: scores below this skip the LLM grading step.
    decline_threshold: float = -5.0


class PricingModelConfig(BaseModel):
    """Per-million-token price for one model."""

    input_per_million: float
    output_per_million: float


class PricingConfig(BaseModel):
    """Pricing for all models.  Extra keys are treated as model-ID entries."""

    model_config = {"extra": "allow"}

    def get_model(self, model_id: str) -> PricingModelConfig | None:
        """Return pricing for *model_id*, or ``None`` if not configured."""
        extra = self.model_extra or {}
        raw = extra.get(model_id)
        if raw is None:
            return None
        return PricingModelConfig(**raw) if isinstance(raw, dict) else raw

    def as_dict(self) -> dict[str, PricingModelConfig]:
        """Return all entries as a plain dict keyed by model ID."""
        extra = self.model_extra or {}
        return {
            k: (PricingModelConfig(**v) if isinstance(v, dict) else v)
            for k, v in extra.items()
        }


class PromptsConfig(BaseModel):
    version: str
    path: str


class PathsConfig(BaseModel):
    chroma_db: str
    bm25_index: str
    documents: str


class ProjectConfig(BaseModel):
    name: str
    version: str


class Settings(BaseModel):
    project: ProjectConfig
    provider: str = "anthropic"  # "anthropic" or "ollama"
    models: ModelsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    prompts: PromptsConfig
    paths: PathsConfig
    pricing: PricingConfig = PricingConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _find_config_file(config_path: str) -> Path:
    """Resolve *config_path* to an absolute Path.

    Search order:
    1. ``CITESAGE_CONFIG`` environment variable (absolute path).
    2. *config_path* if already absolute.
    3. Walk up from ``Path.cwd()`` until the file is found.
    """
    env = os.environ.get("CITESAGE_CONFIG")
    if env:
        return Path(env)

    p = Path(config_path)
    if p.is_absolute():
        return p

    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / config_path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Config file '{config_path}' not found starting from {Path.cwd()}"
    )


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config.yaml") -> Settings:
    """Load and cache settings from *config_path*.

    The result is cached for the lifetime of the process.  Call
    ``get_settings.cache_clear()`` in tests when you need a fresh load.
    """
    path = _find_config_file(config_path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Settings(**data)
