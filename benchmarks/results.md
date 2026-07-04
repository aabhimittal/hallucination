# Benchmark results: Baseline RAG vs VERITAS

- Questions: **40** (20 answerable, 12 unanswerable, 8 adversarial)
- Provider: **mock** (MockLLM, hallucination injection rate 35%, seed 13)
- Judge: model-free lexical-entailment grader against the full corpus (same ruler for both systems). Answering an unanswerable/adversarial question at all counts as a hallucinated question.

| Metric | Baseline RAG | VERITAS | Better |
|---|---|---|---|
| Hallucination rate (per question) | 57.5% | 17.5% | lower |
| Unsupported claim rate | 28.1% | 0.0% | lower |
| Mean groundedness | 61.3% | 100.0% | higher |
| Abstention recall (unanswerable) | 0.0% | 65.0% | higher |
| False abstention rate (answerable) | 0.0% | 0.0% | lower |
| Answer accuracy (answerable) | 90.0% | 90.0% | higher |
| Answer coverage (answerable) | 100.0% | 100.0% | higher |
| Citation precision | — | 95.8% | higher |
| Mean latency (s) | 0.000 | 0.001 | lower |

Reproduce with: `python benchmarks/run_benchmark.py`
