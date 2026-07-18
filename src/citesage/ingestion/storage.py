"""Persistent storage backends for CiteSage.

ChromaStore  — vector store backed by ChromaDB (cosine similarity).
BM25Index    — keyword index backed by rank_bm25, serialised to disk with pickle.

Both backends use chunk_id as the canonical key, enabling upsert semantics
(re-ingesting the same file is idempotent).
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from .models import Chunk


@lru_cache(maxsize=None)
def _load_embedder(name: str) -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer embedder by name.

    Loaded once per process and shared across ``ChromaStore`` instances. The
    graph builds a ``Retriever`` (which constructs a ``ChromaStore``) inside
    node functions, so without this cache the embedder reloaded every query,
    contributing to the OOM that killed eval runs mid-stream. Cache key is the
    config-provided name, preserving ``models.embedder`` swappability.
    """
    return SentenceTransformer(name)


@lru_cache(maxsize=None)
def _get_chroma_client(path: str) -> chromadb.PersistentClient:
    """Create (and cache) a ChromaDB persistent client per path.

    ``retrieve_node`` constructs a fresh ``Retriever`` → ``ChromaStore`` every
    query; without this cache a new PersistentClient was built each time.
    One client per path is shared process-wide — upserts through it are
    immediately visible to queries, so no invalidation is needed.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------


class ChromaStore:
    """Store and query chunk embeddings in ChromaDB.

    Uses cosine distance and upsert semantics so re-ingestion is safe.
    """

    COLLECTION_NAME = "citesage"

    def __init__(self, persist_dir: str | None = None) -> None:
        settings = get_settings()
        path = persist_dir or settings.paths.chroma_db

        self._client = _get_chroma_client(str(path))
        self._embedder = _load_embedder(settings.models.embedder)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk]) -> int:
        """Embed and upsert *chunks*.  Returns the number of chunks processed."""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number or 0,
                    "section_heading": c.section_heading or "",
                    "chunk_index": c.chunk_index,
                    "doc_type": c.doc_type,
                    "ingestion_timestamp": c.ingestion_timestamp,
                    "content_hash": c.content_hash,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (Chunk, distance) pairs for *query_text*.

        Lower distance = more similar (cosine space).
        """
        settings = get_settings()
        k = top_k if top_k is not None else settings.retrieval.vector_top_k
        embedding = self._embedder.encode(
            [query_text], show_progress_bar=False
        ).tolist()

        kwargs: dict = dict(
            query_embeddings=embedding,
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: list[tuple[Chunk, float]] = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = Chunk(
                chunk_id=cid,
                content=doc,
                source_file=meta["source_file"],
                page_number=meta["page_number"] or None,
                section_heading=meta["section_heading"] or None,
                chunk_index=meta["chunk_index"],
                doc_type=meta["doc_type"],
                ingestion_timestamp=meta["ingestion_timestamp"],
                content_hash=meta["content_hash"],
                token_count=meta["token_count"],
            )
            chunks.append((chunk, float(dist)))

        return chunks

    def count(self) -> int:
        """Return total number of chunks stored."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# BM25 keyword index
# ---------------------------------------------------------------------------

# Loaded BM25 indexes keyed by resolved index path, valued as
# (file mtime_ns, instance). ``retrieve_node`` builds a fresh ``Retriever`` →
# ``BM25Index.load()`` every query; without this cache the full pickle was
# deserialized from disk per query. The mtime tag keeps the cache coherent
# even when another process rewrites the index (e.g. CLI ingest while the API
# server is running): a changed mtime forces a reload, at the cost of one
# stat() per query.
_BM25_CACHE: dict[str, tuple[int, "BM25Index"]] = {}


class BM25Index:
    """In-memory BM25Okapi index persisted to disk via pickle.

    Design decisions:
    - Chunks are stored alongside the serialised corpus so that search results
      carry full metadata without a round-trip to ChromaDB.
    - Upsert semantics: adding a chunk whose chunk_id is already present
      replaces the old entry so re-ingestion stays idempotent.
    - Call save() explicitly after add() to persist; this keeps hot-path
      ingestion from doing unnecessary I/O when batching many files.
    """

    def __init__(self, index_path: str | None = None) -> None:
        settings = get_settings()
        self._path = Path(index_path or settings.paths.bm25_index)
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _rebuild(self) -> None:
        if self._chunks:
            corpus = [self._tokenize(c.content) for c in self._chunks]
            self._index = BM25Okapi(corpus)
        else:
            self._index = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        """Add *chunks* to the index, replacing any existing entry with the
        same chunk_id (upsert semantics)."""
        if not chunks:
            return
        # Replace duplicates first
        self._chunks = [
            c for c in self._chunks if c.chunk_id not in {nc.chunk_id for nc in chunks}
        ]
        self._chunks.extend(chunks)
        self._rebuild()

    def save(self) -> None:
        """Serialise the current index and chunk list to disk.

        Also refreshes the process-wide load cache for this path, so
        subsequent ``BM25Index.load()`` calls (e.g. from a query after an
        ingest) return this up-to-date instance instead of stale state.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "corpus": [self._tokenize(c.content) for c in self._chunks],
        }
        with open(self._path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _BM25_CACHE[str(self._path.resolve())] = (
            self._path.stat().st_mtime_ns,
            self,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_path: str | None = None) -> "BM25Index":
        """Load a persisted index from disk, reusing a process-wide cache.

        The cache entry is tagged with the index file's mtime: a matching
        mtime returns the cached instance without touching the pickle, while
        a changed mtime (another process re-ingested) forces a fresh
        deserialize. Returns an empty index if the file does not yet exist
        (never cached, so the index is picked up as soon as one is written).
        """
        instance = cls(index_path)
        key = str(instance._path.resolve())
        if not instance._path.exists():
            _BM25_CACHE.pop(key, None)
            return instance
        mtime = instance._path.stat().st_mtime_ns
        cached = _BM25_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        with open(instance._path, "rb") as fh:
            payload = pickle.load(fh)
        instance._chunks = payload["chunks"]
        corpus = payload["corpus"]
        if corpus:
            instance._index = BM25Okapi(corpus)
        _BM25_CACHE[key] = (mtime, instance)
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (Chunk, BM25 score) pairs, highest score first."""
        settings = get_settings()
        k = top_k if top_k is not None else settings.retrieval.bm25_top_k

        if self._index is None or not self._chunks:
            return []

        scores = self._index.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._chunks[i], float(s)) for i, s in ranked[:k]]

    def chunk_count(self) -> int:
        """Return number of chunks currently indexed."""
        return len(self._chunks)
