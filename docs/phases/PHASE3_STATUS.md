# Phase 3: Status Report

**Status**: ✓ GOLDEN DATASET CREATED

## Deliverables Completed

### 1. Golden Evaluation Dataset
- **File**: `tests/eval/golden_dataset.json`
- **Total QA Pairs**: 65 (target was 50, bonus coverage)
- **All answers manually verified** against source document
- **All chunk references validated** against actual ChromaDB chunks

### 2. Category Distribution
| Category | Count | Coverage |
|----------|-------|----------|
| factual_lookup | 33 | Basic comprehension |
| multi_hop | 7 | Cross-chunk reasoning |
| unanswerable | 10 | Decline accuracy |
| ambiguous | 6 | Query interpretation |
| exact_term | 9 | BM25 retrieval |
| **Total** | **65** | **✓ Complete** |

### 3. Evaluation Plan
- **File**: `docs/phases/PHASE3_EVALUATION.md`
- Comprehensive evaluation pipeline specification
- Target metrics (accuracy ≥85%, citation precision ≥90%)
- Debugging workflows for common failure modes
- Expected outputs (results.json, errors.jsonl)

### 4. Data Quality Assurance
✓ All factual_lookup answers directly quoted from transformer_architecture.md  
✓ Multi_hop questions validated to require actual chunk synthesis  
✓ Unanswerable questions confirmed absent from source documents  
✓ Exact_term questions target specific numbers, names, formulas  
✓ All expected_source_chunks reference real ChromaDB chunk IDs:
  - 931fb959fc8af9821b22dca2a8a2423fa6c7cb64f040b4e59ae02fa88ad077c3
  - 220b4908c37209a845952a33d9239579a6a7591dda7ba43f18552a7858b396a4
  - 39d3dad1620d0f3e510bc59e7eb396062b7c6e8336cab1c62027c6802db8a56a
  - c806516a4a3d57f9fe09f63b2def4880198094d56f7adebff8ff07ccc22dcde4

---

## Next Steps: Phase 3 Execution

### 3a. Build Evaluation Harness
```bash
python -m citesage.evaluation.run_eval \
  --dataset tests/eval/golden_dataset.json \
  --output tests/eval/results.json \
  --errors tests/eval/errors.jsonl
```

### 3b. Run Baseline Evaluation
- Execute all 65 queries through the full pipeline
- Grade answers against golden set
- Generate per-category accuracy report

### 3c. Analyze Results
- Identify systematic failure modes
- Check if any category dips below 80% accuracy
- Flag chunking issues if citation precision < 90%

### 3d. Iterate (if needed)
- Tune retrieval parameters (vector_top_k, rerank_candidates)
- Adjust decline_threshold if decline recall is too low
- Re-run evaluation until metrics meet targets

### 3e. Sign Off
- All metrics ≥ Phase 3 targets
- Error analysis complete
- Ready for Phase 4 (API/UI deployment)

---

## Files Modified/Created This Phase

```
citesage/
├── tests/eval/
│   └── golden_dataset.json          [NEW] 65 QA pairs, manually verified
├── docs/phases/
│   ├── PHASE3_EVALUATION.md         [NEW] Full evaluation spec
│   └── PHASE3_STATUS.md             [NEW] This file
└── README.md                        [PENDING] Add eval workflow docs
```

---

**Completed by**: Claude  
**Timestamp**: 2026-04-01  
**Phase Ready**: YES ✓

---

## Phase 3 Tuning Addendum (2026-04-24)

The initial baseline run on Ollama (qwen3:8b + qwen3-small) missed every
target:

| Metric             | Target | Baseline (Ollama, pre-tuning) |
| ------------------ | ------ | ----------------------------- |
| Accuracy           | ≥ 85 % | 66.9 %                        |
| Citation precision | ≥ 90 % | 23.6 %                        |
| Decline recall    | ≥ 85 % | 50 %                          |
| p95 latency        | < 5 s  | 205.9 s                       |

Root causes identified and fixed in-place (see commits touching
[graph/nodes.py](../../src/citesage/graph/nodes.py),
[evaluation/run_eval.py](../../src/citesage/evaluation/run_eval.py),
[config.yaml](../../config.yaml),
[utils/llm_factory.py](../../src/citesage/utils/llm_factory.py)):

1. **Grade-relevance JSON fallback kept ALL chunks on parse failure**
   → flipped to `[]` so the thorough path declines instead of answering
   from unfiltered candidates. The parser now also handles prose-wrapped
   JSON arrays.
2. **Citation precision was measured as a boolean per-query**
   (`cited ⊆ expected`) then averaged — meaning a query that cited 9
   correct + 1 wrong chunk scored 0 %. Matches the PHASE3 spec's actual
   definition (`mean(|cited ∩ expected| / |cited|`)) now.
3. **Routing thresholds too permissive**: `confidence_threshold`
   0.7 → 0.8, `decline_threshold` -5.0 → -3.0.
4. **Ollama timeouts caused three silent failures**: added an explicit
   `ollama.Client(timeout=60s)` + a distinct `OllamaConnectionError` so
   retries see real errors instead of socket hangs.

The Anthropic-provider green run (required to hit p95 < 5 s) is pending
funded Anthropic credits. Re-run with:

```bash
# set ANTHROPIC_API_KEY, set provider=anthropic in config.yaml
python -m citesage.evaluation.run_eval \
  --dataset tests/eval/golden_dataset.json \
  --output  reports/baseline_scores.json
```

