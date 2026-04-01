"""IngestPipeline — orchestrates load → chunk → store for one file or a directory.

Usage:
    pipeline = IngestPipeline()
    chunks = pipeline.ingest_file("docs/report.pdf")
    # BM25 index is auto-saved after each file.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from .chunker import ChunkerFactory
from .loaders import ALLOWED_EXTENSIONS, load_document
from .models import Chunk
from .storage import BM25Index, ChromaStore

logger = structlog.get_logger(__name__)


class IngestPipeline:
    """Coordinate loading, chunking, and storage for document ingestion.

    Both *chroma_store* and *bm25_index* default to loading from the paths
    defined in config.yaml, making the class directly usable with zero args.
    Pass explicit instances in tests to inject mocks or temp directories.
    """

    def __init__(
        self,
        chroma_store: ChromaStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._store = chroma_store or ChromaStore()
        self._bm25 = bm25_index or BM25Index.load()
        self._factory = ChunkerFactory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str | Path) -> list[Chunk]:
        """Ingest a single file end-to-end.

        Steps:
        1. Load into Document(s) via the appropriate loader.
        2. Chunk each Document with the doc-type-specific TokenChunker.
        3. Upsert all chunks into ChromaDB.
        4. Add all chunks to the BM25 index and persist to disk.

        Returns all Chunk objects produced so callers can inspect them.
        """
        path = Path(file_path)
        log = logger.bind(file=str(path))
        log.info("ingestion.file.start")

        documents = load_document(path)
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunker = self._factory.get_chunker(doc.doc_type)
            all_chunks.extend(chunker.chunk(doc))

        chroma_count = self._store.upsert(all_chunks)
        self._bm25.add(all_chunks)
        self._bm25.save()

        log.info(
            "ingestion.file.complete",
            chunks_stored=chroma_count,
            bm25_total=self._bm25.chunk_count(),
        )
        return all_chunks

    def ingest_directory(self, dir_path: str | Path) -> list[Chunk]:
        """Recursively ingest all supported files under *dir_path*.

        Files are processed in lexicographic order by extension then path
        for deterministic output.  Unsupported file types are silently skipped.
        """
        dir_path = Path(dir_path)
        all_chunks: list[Chunk] = []

        files: list[Path] = []
        for ext in sorted(ALLOWED_EXTENSIONS):
            files.extend(sorted(dir_path.rglob(f"*{ext}")))

        log = logger.bind(directory=str(dir_path), file_count=len(files))
        log.info("ingestion.directory.start")

        for file in files:
            chunks = self.ingest_file(file)
            all_chunks.extend(chunks)

        log.info("ingestion.directory.complete", total_chunks=len(all_chunks))
        return all_chunks

    def rebuild_bm25(self, dir_path: str | Path | None = None) -> None:
        """Re-index everything from ChromaDB into a fresh BM25 index (--rebuild flag).

        If *dir_path* is given, re-ingest from disk instead.
        """

        if dir_path is not None:
            self._bm25 = BM25Index(
                str(self._bm25._path)
            )  # fresh empty index, same path
            self.ingest_directory(dir_path)
            return

        # Pull all chunks stored in Chroma and rebuild BM25 from them
        self._bm25 = BM25Index(str(self._bm25._path))
        offset = 0
        batch = 500
        while True:
            results = self._store._collection.get(
                limit=batch,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = results["ids"]
            if not ids:
                break
            from .models import Chunk

            for doc, meta, cid in zip(results["documents"], results["metadatas"], ids):
                self._bm25._chunks.append(
                    Chunk(
                        chunk_id=cid,
                        content=doc,
                        source_file=meta["source_file"],
                        page_number=meta.get("page_number") or None,
                        section_heading=meta.get("section_heading") or None,
                        chunk_index=meta["chunk_index"],
                        doc_type=meta["doc_type"],
                        ingestion_timestamp=meta["ingestion_timestamp"],
                        content_hash=meta["content_hash"],
                        token_count=meta["token_count"],
                    )
                )
            offset += batch

        self._bm25._rebuild()
        self._bm25.save()
        logger.info("bm25.rebuild.complete", chunks=self._bm25.chunk_count())
