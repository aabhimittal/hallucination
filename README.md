---
title: VERITAS — Hallucination-Reduction RAG
emoji: 🔬
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Claim-verified RAG that measurably reduces hallucination
---

# 🔬 VERITAS — Hallucination-Reduction RAG

**VERITAS** (*Verification-Enhanced Retrieval with Iterative Truth Assessment
and Scoring*) is a RAG pipeline that treats hallucination as something to
**gate, verify, repair, and measure** — not just hope away. It runs on top of
**any LLM** (Claude, GPT, Llama, or the bundled offline mock) with zero heavy
dependencies: retrieval and verification are pure Python.

## The technique

| Stage | What happens | Temperature |
|---|---|---|
| 1. Retrieve | Hybrid BM25 + TF-IDF cosine over sentence-window chunks | — |
| 2. **Gate** | Evidence-confidence threshold → **abstain** instead of guessing | — |
| 3. Generate | Citation-contract prompt: answer *only* from evidence, cite every sentence | 0.1 |
| 4. Decompose | Split the draft into atomic factual claims | 0.0 |
| 5. **Verify** | Two independent judges per claim: model-free lexical entailment (strict number matching) + LLM chain-of-verification | 0.0 |
| 6. **Repair** | Rewrite unsupported claims from evidence or drop them; abstain if most of the draft fails | 0.0 |
| 7. Score | **Groundedness** (fraction of supported claims) ships with every answer | — |

Why it works: most RAG hallucinations are answers to questions the corpus
can't support (killed by the gate); the rest are fabricated details woven into
otherwise-grounded text (caught claim-by-claim by the dual verifier — a
persuasive LLM cannot sweet-talk the lexical judge, and fabricated numbers
light it up instantly).

## Benchmark results

40 questions (20 answerable, 12 unanswerable, 8 adversarial) over a bundled
10-document corpus. Both systems run on the same deterministic `MockLLM`
configured to hallucinate on **35%** of its answers — simulating an unreliable
base model — and are graded by the same model-free lexical judge against the
full corpus. Reproduce: `python benchmarks/run_benchmark.py`.

| Metric | Baseline RAG | VERITAS | Better |
|---|---|---|---|
| Hallucination rate (per question) | 57.5% | **17.5%** | lower |
| Unsupported claim rate | 28.1% | **0.0%** | lower |
| Mean groundedness | 61.3% | **100.0%** | higher |
| Abstention recall (unanswerable) | 0.0% | **65.0%** | higher |
| False abstention rate (answerable) | 0.0% | **0.0%** | lower |
| Answer accuracy (answerable) | 90.0% | **90.0%** | higher |
| Citation precision | — | **95.8%** | higher |

VERITAS removes **100% of fabricated claims** from delivered answers and cuts
question-level hallucination by **~70%**, with **no loss of accuracy or
coverage** on answerable questions. (Residual "hallucinations" are strictly-
scored cases where the system answered an adversarial question with true-but-
irrelevant corpus facts.) Benchmark a live model with
`python benchmarks/run_benchmark.py --provider anthropic|openai|hf`.

## Quickstart

```python
# pip install -e .            (pure stdlib; extras: [anthropic], [openai], [hf], [demo])
from veritas import Document, HybridRetriever, MockLLM, VeritasPipeline, chunk_corpus

docs = [Document("d1", "Mount Everest is the highest mountain on Earth. "
                       "Its summit stands at 8849 meters above sea level.")]
retriever = HybridRetriever(chunk_corpus(docs))

llm = MockLLM()                       # offline demo model
# from veritas import AnthropicClient; llm = AnthropicClient()   # or any real LLM

result = VeritasPipeline(llm, retriever).answer("How tall is Mount Everest?")
print(result.answer)          # "Its summit stands at 8849 meters above sea level. [c1]"
print(result.groundedness)    # 1.0
print(result.abstained)       # False — and True (with reason) for unanswerable questions
for verdict in result.final_verdicts:
    print(verdict.label, verdict.claim.text)
```

## Demo

The Gradio demo (this Space) runs keyless out of the box on the deterministic
mock model — flip the provider dropdown to Anthropic / OpenAI-compatible /
Hugging Face Inference and paste your own API key (used per-request, never
stored) to drive it with a real LLM. It shows the VERITAS answer next to the
baseline RAG answer, per-claim verdicts, the full pipeline trace, and the
benchmark charts.

Run locally: `pip install -r requirements.txt && python app.py`

Deploy your own Space: `python scripts/deploy_space.py --repo <user>/veritas-demo`
(needs a Hugging Face write token via `--token` or the `HF_TOKEN` env var).

## Repository layout

```
src/veritas/          the pipeline (chunking, retrieval, llm adapters, prompts,
                      claims, verification, pipeline, metrics)
tests/                62 offline tests — pytest
benchmarks/           corpus + 40-question dataset + runner + committed results
skills/hallucination-reduction/SKILL.md
                      reusable playbook: prompting, chain-of-verification,
                      temperature settings, RAG design for ANY LLM
app.py                Gradio demo (Hugging Face Spaces entrypoint)
scripts/deploy_space.py
                      one-command Space deployment
```

## The skill

[`skills/hallucination-reduction/SKILL.md`](skills/hallucination-reduction/SKILL.md)
distills the technique into a provider-agnostic playbook — grounding-contract
prompting, chain-of-thought vs chain-of-verification, a per-task temperature
table (including models that reject sampling params), and the RAG design
checklist. Drop the `skills/` folder into a Claude Code project (or any
agent-skills-compatible harness) and it activates whenever you work on
hallucination-sensitive LLM features.

## Testing

```bash
pip install -e ".[dev]"
pytest        # 62 tests, all offline, < 1s
```

## License

MIT
