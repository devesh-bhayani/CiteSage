# Phase 3: Evaluation and Testing (Week 3)

## Goal: Golden dataset → evaluation metrics → system validation

Phase 3 focuses on rigorous evaluation of the CiteSage system against manually-verified golden answers to ensure the RAG pipeline produces correct, cited responses.

## 1. Golden Evaluation Dataset

### Location
- File: `tests/eval/golden_dataset.json`
- Schema:
  ```json
  {
    "id": "eval_001",
    "question": "...",
    "expected_answer": "...",
    "expected_source_chunks": ["chunk_id_1", "chunk_id_2"],
    "category": "factual_lookup|multi_hop|unanswerable|ambiguous|exact_term",
    "difficulty": "easy|medium|hard",
    "decline_reason": "..."  // for unanswerable only
  }
  ```

### Dataset Composition (50 pairs)

| Category | Count | Purpose |
|----------|-------|---------|
| **factual_lookup** | 20 | Straightforward questions answerable from single chunks |
| **multi_hop** | 10 | Questions requiring synthesis of 2+ chunks |
| **unanswerable** | 10 | Questions the system SHOULD decline with confidence |
| **ambiguous** | 5 | Vague questions testing query understanding |
| **exact_term** | 5 | Specific names/codes/numbers testing BM25 matching |

### Quality Requirements
- ✓ Every `expected_answer` must be manually verifiable against source documents
- ✓ `expected_source_chunks` must reference chunks that actually exist in ChromaDB
- ✓ Unanswerable questions must be genuinely absent from docs (not just hard to find)
- ✓ Ambiguous questions should have reasonable multi-path answers

---

## 2. Evaluation Metrics

### Per-Query Metrics
- **Answer Correctness**: Does the LLM answer match `expected_answer`? (semantic equivalence via embedding similarity or Claude grading)
- **Citation Accuracy**: Are the cited sources in `expected_source_chunks`?
- **Decline Accuracy**: Does the system decline unanswerable questions?
- **Confidence Calibration**: Is the decline threshold well-calibrated?

### Aggregate Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | ≥90% | % of queries with correct answer |
| **Citation Precision** | ≥95% | % of cited sources that are relevant |
| **Decline Recall** | ≥90% | % of unanswerable questions declined |
| **Decline Precision** | ≥85% | % of declines that were justified |
| **Latency (p50/p95)** | <2s / <5s | Query → answer time |
| **Token Usage (per query)** | <2000 | Total tokens (input + output) |

### Category Breakdown
- Report accuracy separately for each category (factual_lookup, multi_hop, etc.)
- Identify systematic failure modes (e.g. "multi_hop fails 30% — needs better retrieval")

---

## 3. Evaluation Pipeline

### Step 1: Load Golden Dataset
```python
import json

with open("tests/eval/golden_dataset.json") as f:
    dataset = json.load(f)

# Filter by category if needed
factual = [q for q in dataset if q["category"] == "factual_lookup"]
unanswerable = [q for q in dataset if q["category"] == "unanswerable"]
```

### Step 2: Run Query-Answer Pipeline
For each question, execute the full CiteSage pipeline:
```python
from citesage.graph.pipeline import build_rag_pipeline

pipeline = build_rag_pipeline()

for qa in dataset:
    result = pipeline.invoke({"question": qa["question"]})
    # result = {
    #   "answer": "...",
    #   "sources": [chunk_id, ...],
    #   "confidence": 0.95,
    #   "declined": False,
    #   "decline_reason": None
    # }
```

### Step 3: Grade Answers
Use Claude Haiku to semantically grade each answer:
```python
# For factual correctness:
# - Compare result["answer"] vs expected_answer
# - Check semantic equivalence (embedding distance or LLM grading)

# For citations:
# - Check if result["sources"] ⊆ expected_source_chunks
# - Evaluate source relevance via cross-encoder scores

# For declining:
# - Check if result["declined"] == True for unanswerable questions
# - Check if result["declined"] == False for answerable questions
```

### Step 4: Aggregate & Report
```python
# Compute accuracy, precision, recall by category
# Identify failure patterns (e.g., retrieval misses, generation errors)
# Log detailed error cases for debugging
```

---

## 4. Expected Outputs

### evaluation_results.json
```json
{
  "timestamp": "2026-04-01T20:00:00Z",
  "total_queries": 50,
  "by_category": {
    "factual_lookup": {
      "count": 20,
      "correct": 19,
      "accuracy": 0.95,
      "failures": [...]
    },
    "multi_hop": {
      "count": 10,
      "correct": 8,
      "accuracy": 0.80,
      "failures": [...]
    },
    ...
  },
  "aggregate": {
    "accuracy": 0.88,
    "citation_precision": 0.94,
    "decline_recall": 0.92,
    "decline_precision": 0.88
  },
  "latency": {
    "p50_ms": 1250,
    "p95_ms": 3800
  },
  "token_usage": {
    "total_input_tokens": 85000,
    "total_output_tokens": 12000,
    "average_per_query": 1940
  }
}
```

### evaluation_errors.jsonl
One line per failure, with detailed context:
```jsonl
{"id": "eval_042", "category": "multi_hop", "error": "answer_mismatch", "expected": "...", "got": "...", "debug": {...}}
```

---

## 5. Phase Completion Criteria

- ✓ Golden dataset: 50 QA pairs, all manually verified
- ✓ Evaluation script: runs queries, grades answers, outputs metrics
- ✓ Aggregate accuracy ≥ 85% (baseline for Phase 3 exit)
- ✓ Citation precision ≥ 90%
- ✓ Decline recall ≥ 85%
- ✓ All test categories covered with per-category analysis
- ✓ Error log: <5 major systematic failures (anything ≥3% of category)

### Definition of "correct answer"
- Semantic match to expected_answer (within 0.85 embedding similarity or LLM passes grading)
- OR factually equivalent (e.g., different phrasing but same meaning)
- Must not hallucinate facts unsupported by sources

### Definition of "good citations"
- All cited sources are in expected_source_chunks (may be subset)
- Reranker confidence ≥ threshold
- No false citations (never cite irrelevant chunks)

---

## 6. Debugging Workflows

### "System is declining too much"
→ Check `decline_threshold` and `confidence_threshold` in config.yaml
→ Lower thresholds to accept more borderline cases

### "Multi-hop questions fail"
→ Check retrieval: are both relevant chunks being retrieved? (vector_top_k + BM25)
→ Check reranker: is it combining chunks effectively?
→ Consider increasing rerank_candidates or rerank_top_k

### "Citations are incorrect"
→ Verify expected_source_chunks in golden dataset (manually re-check doc)
→ Check cross-encoder scores: is reranker confident in its choices?
→ May indicate document chunking is too aggressive/conservative

### "Latency is high (>5s)"
→ Profile which component is slowest: retrieval, reranking, or generation?
→ May need to reduce vector_top_k or rerank_candidates

---

## 7. Next Steps (Phase 4)

- Deployment: API + Streamlit UI
- User feedback loops: collect eval queries from production
- Iterative improvement: retrain embeddings, fine-tune thresholds based on real queries
- Monitoring: log all queries, track declining rate drift over time

