"""Document loaders for PDF, Markdown, HTML, and plain text.

Safety constraints (from ingestion CLAUDE.md):
- Whitelist: .pdf, .md, .html, .txt only.
- Max file size: 50 MB.
- PDF parsing is sandboxed: all fitz errors are caught and re-raised as ValueError.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from markdownify import markdownify

from .models import Document

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".html", ".txt"})
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


def _validate_file(path: Path) -> None:
    """Raise ValueError if *path* is an unsupported type or too large."""
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}"
        )
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {size / 1_048_576:.1f} MB (limit 50 MB): {path}"
        )


class PDFLoader:
    """Load a PDF into one Document per page using PyMuPDF.

    Empty pages (whitespace only) are skipped.
    Section heading is heuristically derived from the largest font span on the page.
    """

    def load(self, file_path: str | Path) -> list[Document]:
        """Return a list of page-level Documents."""
        path = Path(file_path)
        _validate_file(path)

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{path}': {exc}") from exc

        documents: list[Document] = []
        try:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if not text.strip():
                    continue
                documents.append(
                    Document(
                        content=text,
                        source_file=str(path),
                        doc_type="pdf",
                        page_number=page_num,
                        section_heading=self._largest_span(page),
                    )
                )
        finally:
            pdf.close()

        return documents

    @staticmethod
    def _largest_span(page: fitz.Page) -> str | None:
        """Return the text of the largest font span on *page* as a heading proxy."""
        max_size = 0.0
        heading: str | None = None
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size: float = span.get("size", 0.0)
                    text: str = span.get("text", "").strip()
                    if text and size > max_size:
                        max_size = size
                        heading = text
        return heading


class MarkdownLoader:
    """Load a Markdown file as a single Document."""

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        _validate_file(path)
        content = path.read_text(encoding="utf-8")

        heading: str | None = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                break

        return [
            Document(
                content=content,
                source_file=str(path),
                doc_type="markdown",
                section_heading=heading,
            )
        ]


class HTMLLoader:
    """Load an HTML file, converting to readable Markdown text."""

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        _validate_file(path)
        raw_html = path.read_text(encoding="utf-8")
        content = markdownify(raw_html, strip=["script", "style"])

        return [
            Document(
                content=content,
                source_file=str(path),
                doc_type="html",
            )
        ]


class TextLoader:
    """Load a plain text file as a single Document."""

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        _validate_file(path)
        content = path.read_text(encoding="utf-8")

        return [
            Document(
                content=content,
                source_file=str(path),
                doc_type="txt",
            )
        ]


# ---------------------------------------------------------------------------
# Auto-dispatch helper
# ---------------------------------------------------------------------------

_LOADERS: dict[str, PDFLoader | MarkdownLoader | HTMLLoader | TextLoader] = {
    ".pdf": PDFLoader(),
    ".md": MarkdownLoader(),
    ".html": HTMLLoader(),
    ".txt": TextLoader(),
}


def load_document(file_path: str | Path) -> list[Document]:
    """Detect file type and return a list of Documents.

    Raises ValueError for unsupported extensions or files exceeding 50 MB.
    """
    path = Path(file_path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"No loader registered for extension '{path.suffix}'. "
            f"Supported: {sorted(_LOADERS)}"
        )
    return loader.load(path)
