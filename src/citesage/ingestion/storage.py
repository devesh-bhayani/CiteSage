"""Persistent storage backends for CiteSage.

ChromaStore  — vector store backed by ChromaDB (cosine similarity).
BM25Index    — keyword index backed by rank_bm25, serialised to disk with pickle.

Both backends use chunk_id as the canonical key, enabling upsert semantics
(re-ingesting the same file is idempotent).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from .models import Chunk


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
        Path(path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=path)
        self._embedder = SentenceTransformer(settings.models.embedder)
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
        """Serialise the current index and chunk list to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "corpus": [self._tokenize(c.content) for c in self._chunks],
        }
        with open(self._path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_path: str | None = None) -> "BM25Index":
        """Load a persisted index from disk.  Returns an empty index if the
        file does not yet exist."""
        instance = cls(index_path)
        if not instance._path.exists():
            return instance
        with open(instance._path, "rb") as fh:
            payload = pickle.load(fh)
        instance._chunks = payload["chunks"]
        corpus = payload["corpus"]
        if corpus:
            instance._index = BM25Okapi(corpus)
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
