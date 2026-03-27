# Ingestion Rules
## Chunking — TOKEN-BASED (Critical)
Use tiktoken length_function with RecursiveCharacterTextSplitter:
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100,
    length_function=lambda text: len(enc.encode(text)),
    separators=["\n\n", "\n", ". ", " "]
)
```
DO NOT pass chunk_size without length_function — defaults to characters.
Chunking config is per doc_type in config.yaml. Build a ChunkerFactory.
Metadata: source_file, page_number, section_heading, chunk_index, doc_type, ingestion_timestamp, content_hash.
Idempotency: SHA-256(chunk_text + source_file) as ChromaDB ID.
BM25: serialize with pickle after ingestion. --rebuild flag for re-indexing.
Upload safety: whitelist PDF/MD/HTML/TXT only. Max 50MB. Sandbox PDF parsing.
