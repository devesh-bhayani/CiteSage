# Evaluation Rules
Four layers: (1) golden dataset + RAGAS for known cases, (2) VCR for determinism, (3) LLM-judge (supplement), (4) human review monthly on 20 queries.
Golden dataset: 50+ QA pairs, 5 categories. At least 10 unanswerable.
CI hard gates: unit tests pass, no crashes, decline works. RAGAS as trends, not pass/fail.
Weekly cron: fresh eval (non-VCR). Monthly: human review of sampled queries.
Cost: Haiku for eval grading. Budget cap $2 per eval run.
