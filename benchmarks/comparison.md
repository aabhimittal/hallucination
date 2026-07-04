# Technique comparison

- Questions: **40** (20 answerable, 12 unanswerable, 8 adversarial)
- Provider: **mock** (MockLLM, 35% injected hallucinations, seed 13)
- Same corpus, same independent lexical judge for every technique. ↑ = higher is better, ↓ = lower is better.

| Metric | Baseline RAG | VERITAS | Semantic Entropy | Quote Grounding | Multi-Agent Consensus | Neurosymbolic Guardrails | Calibrated Selective Prediction | Graph-RAG |
|---|---|---|---|---|---|---|---|---|
| Hallucination rate ↓ | 72.5% | 17.5% | 12.5% | 17.5% | 17.5% | 22.5% | 25.0% | 25.0% |
| Unsupported claim rate ↓ | 33.8% | 0.0% | 0.0% | 0.0% | 0.0% | 3.9% | 9.6% | 9.1% |
| Mean groundedness ↑ | 55.0% | 100.0% | 100.0% | 100.0% | 100.0% | 97.5% | 90.7% | 87.2% |
| Abstention recall ↑ | 0.0% | 65.0% | 75.0% | 65.0% | 65.0% | 65.0% | 65.0% | 70.0% |
| False abstention rate ↓ | 0.0% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 10.0% |
| Answer accuracy ↑ | 80.0% | 90.0% | 90.0% | 90.0% | 80.0% | 90.0% | 90.0% | 83.3% |
| Answer coverage ↑ | 100.0% | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% |
| Calibration ECE ↓ | — | 0.354 | 0.377 | — | — | 0.391 | 0.522 | 0.429 |
| Calibration AUROC ↑ | — | 0.534 | 0.354 | — | — | 0.461 | 0.266 | 0.436 |
| Risk–coverage AURC ↓ | — | 0.154 | 0.401 | — | — | 0.230 | 0.447 | 0.405 |
| Mean latency (s) ↓ | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 |

**Families:** Baseline RAG (baseline), Calibrated Selective Prediction (uncertainty), Graph-RAG (graph), Multi-Agent Consensus (verify), Neurosymbolic Guardrails (guardrail), Quote Grounding (verify), Semantic Entropy (uncertainty), VERITAS (verify)

Notes: Semantic Entropy and the abstaining techniques trade coverage for safety — read hallucination rate together with false-abstention rate. Graph-RAG matches flat retrieval on single-hop factoids; its multi-hop advantage needs a reasoning LLM and relational corpus (the offline mock answers by lexical overlap). DoLa (white-box) is compared separately — see `benchmarks/dola.md` / `run_dola.py`.

Reproduce: `python benchmarks/run_comparison.py`
