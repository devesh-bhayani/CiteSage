# Generation Rules
Two-tier: FAST PATH (reranker score >= 0.7): retrieve → rerank → generate. 1 LLM call.
THOROUGH PATH (score < 0.7): + grade + verify + optional rewrite. 3-4 LLM calls.
Citation verification — hybrid: (1) deterministic token overlap check, (2) LLM judge for weak only, (3) decide.
Decline when >50% unsupported. Generation: Sonnet. Routing/grading/verify: Haiku.
All LLM calls: retry with backoff (max 3, base 1s). Log tokens per call.
Prompts: YAML files in prompts/ only. Never inline.
