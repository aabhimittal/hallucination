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

## Technique zoo — quantitative comparison

Beyond VERITAS, the repo implements a spread of hallucination-reduction
techniques from the recent literature behind **one comparable interface**
(`veritas.techniques`), and benchmarks them head-to-head on the same corpus
with the same independent judge. Run `python benchmarks/run_comparison.py`.

| Technique | Family | Idea | Runtime |
|---|---|---|---|
| **Baseline RAG** | baseline | Retrieve, stuff context, answer at T=0.7, trust output | any LLM |
| **VERITAS** | verify | Gate → cite → decompose → dual-judge verify → repair → score | any LLM |
| **Semantic Entropy** | uncertainty | Sample N, cluster by meaning, abstain on high entropy (Nature 2024) | any LLM |
| **Quote Grounding** | verify | Extract verbatim quotes, drop any that aren't exact substrings, synthesize from those | any LLM |
| **Multi-Agent Consensus** | verify | Researcher → editor → NLI judge, rewrite on contradiction | any LLM |
| **Neurosymbolic Guardrails** | guardrail | Programmatic input/output rails (scope, citations, no speculation) | any LLM |
| **Calibrated Selective Prediction** | uncertainty | Verbalized confidence + ECE/AUROC/risk–coverage; withhold low-confidence answers | any LLM |
| **Graph-RAG** | graph | Entity/relation graph + multi-hop traversal retrieval | any LLM |
| **DoLa** | decoding | Contrast late vs early transformer layers to amplify facts (Chuang 2023) | **local HF model** |

Representative result (40 questions, MockLLM with 35% injected hallucinations,
same lexical judge for all — `benchmarks/comparison.md`):

| Metric | Baseline | VERITAS | Sem. Entropy | Quote Gr. | Multi-Agent | Guardrails | Calib. | Graph-RAG |
|---|---|---|---|---|---|---|---|---|
| Hallucination rate ↓ | 72.5% | 17.5% | **12.5%** | 17.5% | 17.5% | 22.5% | 25.0% | 25.0% |
| Mean groundedness ↑ | 55.0% | **100%** | **100%** | **100%** | **100%** | 97.5% | 90.7% | 87.2% |
| Abstention recall ↑ | 0% | 65% | **75%** | 65% | 65% | 65% | 65% | 70% |
| False abstention ↓ | **0%** | **0%** | 50% | **0%** | **0%** | **0%** | **0%** | 10% |
| Answer accuracy ↑ | 80% | **90%** | **90%** | **90%** | 80% | **90%** | **90%** | 83% |

The comparison is deliberately honest about tradeoffs: **Semantic Entropy** buys
the lowest hallucination rate by abstaining aggressively (50% false-abstention,
half the answerable coverage) — a *reliability detector* that distrusts a noisy
base model; the **verify** family (VERITAS / Quote Grounding / Multi-Agent) hits
the sweet spot (zero fabricated claims delivered, no false abstention, full
accuracy); **Calibration** exposes the base model's over-confidence via ECE/AUROC
rather than fixing it. There is no single winner — that's the point.

### Run the comparison on a real model — free, on a Colab GPU

No API keys or credits needed: run any open-weights instruct model on Colab's
free T4 and point the whole benchmark at it.
[`notebooks/colab_live_benchmark.ipynb`](notebooks/colab_live_benchmark.ipynb)
does it end-to-end (defaults to `Qwen/Qwen2.5-7B-Instruct`, 4-bit, which fits a
T4). Locally with a GPU:

```bash
pip install -e ".[local]"
python benchmarks/run_comparison.py --provider local \
    --model Qwen/Qwen2.5-7B-Instruct --load-4bit --balanced 5 --out comparison_live
```

`--provider local` uses `veritas.local.LocalChatClient` (applies the model's
chat template + optional 4-bit quantization). Paid alternatives:
`--provider openai --model gpt-4o-mini` (~15¢) or `--provider anthropic|hf`.

**White-box (DoLa)** is compared separately because it needs logit access
(`python benchmarks/run_dola.py`, requires `pip install 'veritas-rag[local]'`).
A real run on gpt2 (`benchmarks/dola.md`): DoLa lifts answer accuracy 0% → 12.5%
and groundedness 2% → 12% over vanilla greedy decoding on the same model, at ~2×
latency — the expected *direction* (DoLa's full effect needs a larger
instruction-tuned model on a factuality benchmark; gpt2 is a wiring check).

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
                      claims, verification, pipeline, metrics, graph)
src/veritas/techniques/
                      the technique zoo behind one interface: semantic_entropy,
                      quote_grounding, multi_agent, guardrails, calibration,
                      graph_rag, decoding (DoLa), nli, wrappers
tests/                95 offline tests — pytest
benchmarks/           corpus + 40-question dataset + runners (run_benchmark,
                      run_comparison, run_dola) + committed results
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
pytest        # 95 tests; add [local] (torch+transformers) to run the DoLa test
```

## License

MIT
