"""VERITAS demo — hallucination-reduction RAG on Hugging Face Spaces.

Runs fully keyless by default (deterministic MockLLM). Users can optionally
plug in their own Anthropic / OpenAI-compatible / Hugging Face Inference key;
keys are used for the request only and never stored.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))  # run without installing the package

import gradio as gr
import pandas as pd

from veritas import (
    BaselineRAG,
    HybridRetriever,
    MockLLM,
    VeritasPipeline,
    chunk_corpus,
    documents_from_texts,
)
from veritas.chunking import load_documents_from_dir
from veritas.graph import GraphRetriever
from veritas.techniques import build_registry
from veritas.techniques.graph_rag import GraphRAGTechnique
from veritas.verification import Verdict

BUNDLED_DOCS = load_documents_from_dir(HERE / "benchmarks" / "corpus")
BUNDLED_CHUNKS = chunk_corpus(BUNDLED_DOCS)
BUNDLED_RETRIEVER = HybridRetriever(BUNDLED_CHUNKS)
BUNDLED_GRAPH_RETRIEVER = GraphRetriever(BUNDLED_CHUNKS)

PROVIDERS = [
    "Demo mode — MockLLM, no key needed",
    "Demo mode — MockLLM with 35% injected hallucinations",
    "Anthropic (Claude)",
    "OpenAI-compatible",
    "Hugging Face Inference",
]

EXAMPLE_QUESTIONS = [
    "What is the height of the summit of Mount Everest in meters?",
    "How many eggs can a queen honey bee lay in a single day?",
    "Who won the 1994 FIFA World Cup?",
    "In what year was the Eiffel Tower demolished?",
]

VERDICT_EMOJI = {
    Verdict.SUPPORTED: "✅",
    Verdict.PARTIAL: "🟡",
    Verdict.UNSUPPORTED: "❌",
}


def build_llm(provider: str, api_key: str, model: str):
    if provider == PROVIDERS[0]:
        return MockLLM()
    if provider == PROVIDERS[1]:
        return MockLLM(hallucination_rate=0.35)
    if not api_key:
        raise gr.Error("This provider needs an API key (or pick a Demo mode).")
    if provider == PROVIDERS[2]:
        from veritas import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-4-8", api_key=api_key)
    if provider == PROVIDERS[3]:
        from veritas import OpenAICompatClient

        return OpenAICompatClient(model=model or "gpt-4o-mini", api_key=api_key)
    from veritas import HFInferenceClient

    return HFInferenceClient(
        model=model or "meta-llama/Llama-3.1-8B-Instruct", token=api_key
    )


def build_retriever(custom_docs: str):
    text = (custom_docs or "").strip()
    if not text:
        return BUNDLED_RETRIEVER, BUNDLED_CHUNKS
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs = documents_from_texts(paragraphs)
    chunks = chunk_corpus(docs)
    return HybridRetriever(chunks), chunks


def _verdict_rows(verdicts):
    return [
        [
            VERDICT_EMOJI.get(v.label, "?") + " " + v.label.value,
            v.claim.text,
            ", ".join(v.claim.citations) or "—",
            f"{v.lexical_score:.2f}",
            v.llm_verdict.value if v.llm_verdict else "—",
        ]
        for v in verdicts
    ]


def ask(question, provider, api_key, model, custom_docs):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Please enter a question.")
    llm = build_llm(provider, api_key, model)
    retriever, _chunks = build_retriever(custom_docs)

    t0 = time.perf_counter()
    veritas_result = VeritasPipeline(llm, retriever).answer(question)
    veritas_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    baseline_result = BaselineRAG(llm, retriever).answer(question)
    baseline_s = time.perf_counter() - t0

    status = "🛑 Abstained" if veritas_result.abstained else "✅ Answered"
    veritas_md = (
        f"### VERITAS answer  \n{veritas_result.answer}\n\n"
        f"**{status}** · evidence confidence **{veritas_result.confidence:.2f}** · "
        f"groundedness **{veritas_result.groundedness:.0%}** · "
        f"{veritas_result.repaired} repaired / {veritas_result.removed} removed · "
        f"{veritas_s * 1000:.0f} ms"
    )
    if veritas_result.abstain_reason:
        veritas_md += f" · reason: *{veritas_result.abstain_reason}*"

    baseline_md = (
        f"### Baseline RAG answer  \n{baseline_result.answer}\n\n"
        f"*No verification, no abstention gate, temperature 0.7 · "
        f"{baseline_s * 1000:.0f} ms*"
    )

    # pipeline trace
    trace_lines = ["### Pipeline trace"]
    for record in veritas_result.trace:
        trace_lines.append(f"- **{record.stage}** — {record.detail}")
    if veritas_result.draft and veritas_result.draft != veritas_result.answer:
        trace_lines.append(f"\n**Draft before verification:** {veritas_result.draft}")
    trace_md = "\n".join(trace_lines)

    verdicts = veritas_result.final_verdicts or veritas_result.draft_verdicts
    claims_df = pd.DataFrame(
        _verdict_rows(verdicts),
        columns=["verdict", "claim", "citations", "lexical score", "LLM verdict"],
    )

    evidence_md = "### Retrieved evidence\n" + "\n".join(
        f"- `[{sc.chunk.chunk_id}]` (score {sc.score:.2f}) {sc.chunk.text}"
        for sc in veritas_result.retrieved
    )
    return veritas_md, baseline_md, trace_md, claims_df, evidence_md


def compare_techniques(question, provider, api_key, model, custom_docs):
    """Run every technique on one question, side by side."""
    question = (question or "").strip()
    if not question:
        raise gr.Error("Please enter a question.")
    llm = build_llm(provider, api_key, model)
    if (custom_docs or "").strip():
        retriever, chunks = build_retriever(custom_docs)
        graph_retriever = GraphRetriever(chunks)
    else:
        retriever, graph_retriever = BUNDLED_RETRIEVER, BUNDLED_GRAPH_RETRIEVER

    techniques = build_registry(llm, retriever)
    techniques[GraphRAGTechnique.name] = GraphRAGTechnique(llm, graph_retriever)

    rows = []
    for name, tech in techniques.items():
        t0 = time.perf_counter()
        res = tech.answer(question)
        ms = (time.perf_counter() - t0) * 1000
        rows.append([
            name,
            tech.family,
            "🛑 abstain" if res.abstained else "✅ answer",
            "—" if res.confidence is None else f"{res.confidence:.2f}",
            (res.answer[:90] + "…") if len(res.answer) > 90 else res.answer,
            f"{ms:.0f}",
        ])
    return pd.DataFrame(
        rows,
        columns=["technique", "family", "decision", "confidence", "answer", "ms"],
    )


def load_comparison():
    md = (HERE / "benchmarks" / "comparison.md").read_text()
    data = json.loads((HERE / "benchmarks" / "comparison.json").read_text())["results"]
    rows = []
    chart_metrics = [
        ("hallucination_rate", "Hallucination rate"),
        ("mean_groundedness", "Mean groundedness"),
        ("abstention_recall", "Abstention recall"),
        ("false_abstention_rate", "False abstention"),
    ]
    for name, entry in data.items():
        for key, label in chart_metrics:
            value = entry["metrics"].get(key)
            if value is not None:
                rows.append({"technique": name, "metric": label,
                             "value": round(value * 100, 1)})
    return md, pd.DataFrame(rows)


def load_benchmark():
    results_md = (HERE / "benchmarks" / "results.md").read_text()
    data = json.loads((HERE / "benchmarks" / "results.json").read_text())
    rows = []
    chart_metrics = [
        ("hallucination_rate", "Hallucination rate"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
        ("mean_groundedness", "Mean groundedness"),
        ("abstention_recall", "Abstention recall"),
        ("answer_accuracy", "Answer accuracy"),
    ]
    for key, label in chart_metrics:
        for system in ("baseline", "veritas"):
            value = data[system].get(key)
            if value is not None:
                rows.append(
                    {"metric": label, "system": system, "value": round(value * 100, 1)}
                )
    return results_md, pd.DataFrame(rows)


with gr.Blocks(title="VERITAS — Hallucination-Reduction RAG") as demo:
    gr.Markdown(
        "# 🔬 VERITAS — Hallucination-Reduction RAG\n"
        "*Verification-Enhanced Retrieval with Iterative Truth Assessment and "
        "Scoring.* Every answer is retrieved with a confidence gate, generated "
        "under a citation contract at low temperature, decomposed into atomic "
        "claims, verified claim-by-claim (lexical entailment + LLM "
        "chain-of-verification), repaired, and scored for groundedness — on "
        "top of **any** LLM."
    )
    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    placeholder=EXAMPLE_QUESTIONS[0],
                )
                gr.Examples(EXAMPLE_QUESTIONS, inputs=question, label="Try these")
                custom_docs = gr.Textbox(
                    label="Optional: your own documents (one per blank-line-separated paragraph; leave empty to use the bundled 10-document corpus)",
                    lines=4,
                )
            with gr.Column(scale=2):
                provider = gr.Dropdown(PROVIDERS, value=PROVIDERS[1], label="LLM provider")
                api_key = gr.Textbox(
                    label="API key (only for real providers; never stored)",
                    type="password",
                )
                model = gr.Textbox(
                    label="Model id (optional; provider default used if empty)"
                )
        ask_btn = gr.Button("Ask", variant="primary")
        with gr.Row():
            veritas_out = gr.Markdown()
            baseline_out = gr.Markdown()
        with gr.Accordion("Claim-level verdicts", open=True):
            claims_out = gr.Dataframe(interactive=False)
        with gr.Accordion("Pipeline trace", open=False):
            trace_out = gr.Markdown()
        with gr.Accordion("Retrieved evidence", open=False):
            evidence_out = gr.Markdown()
        ask_btn.click(
            ask,
            inputs=[question, provider, api_key, model, custom_docs],
            outputs=[veritas_out, baseline_out, trace_out, claims_out, evidence_out],
        )
        question.submit(
            ask,
            inputs=[question, provider, api_key, model, custom_docs],
            outputs=[veritas_out, baseline_out, trace_out, claims_out, evidence_out],
        )

    with gr.Tab("Compare techniques"):
        gr.Markdown(
            "### Run every hallucination-reduction technique on one question\n"
            "Baseline RAG, VERITAS, Semantic Entropy, Quote Grounding, "
            "Multi-Agent Consensus, Neurosymbolic Guardrails, Calibrated "
            "Selective Prediction, and Graph-RAG — same question, same "
            "evidence, side by side."
        )
        with gr.Row():
            cmp_question = gr.Textbox(label="Question", scale=3,
                                      placeholder=EXAMPLE_QUESTIONS[0])
            cmp_provider = gr.Dropdown(PROVIDERS, value=PROVIDERS[1],
                                       label="LLM provider", scale=2)
        with gr.Row():
            cmp_key = gr.Textbox(label="API key (real providers only; never stored)",
                                 type="password")
            cmp_model = gr.Textbox(label="Model id (optional)")
        cmp_docs = gr.Textbox(
            label="Optional: your own documents (blank-line-separated; empty = bundled corpus)",
            lines=2,
        )
        gr.Examples(EXAMPLE_QUESTIONS, inputs=cmp_question, label="Try these")
        cmp_btn = gr.Button("Compare all techniques", variant="primary")
        cmp_out = gr.Dataframe(interactive=False)
        cmp_btn.click(compare_techniques,
                      inputs=[cmp_question, cmp_provider, cmp_key, cmp_model, cmp_docs],
                      outputs=cmp_out)
        cmp_question.submit(compare_techniques,
                            inputs=[cmp_question, cmp_provider, cmp_key, cmp_model, cmp_docs],
                            outputs=cmp_out)

    with gr.Tab("Benchmarks"):
        gr.Markdown("## VERITAS vs Baseline RAG")
        results_md_text, chart_df = load_benchmark()
        gr.Markdown(results_md_text)
        gr.BarPlot(
            chart_df, x="metric", y="value", color="system",
            title="Baseline RAG vs VERITAS (%, higher is better except the two rates)",
            y_lim=[0, 100],
        )
        gr.Markdown("---\n## All techniques compared")
        comparison_md, comparison_df = load_comparison()
        gr.Markdown(comparison_md)
        gr.BarPlot(
            comparison_df, x="technique", y="value", color="metric",
            title="Every technique on the shared benchmark (%)",
            y_lim=[0, 100],
        )
        gr.Markdown(
            "Reproduce: `python benchmarks/run_comparison.py` (all techniques) · "
            "`python benchmarks/run_dola.py` (white-box DoLa, needs a local model) · "
            "add `--provider anthropic|openai|hf` to benchmark a live model."
        )

    with gr.Tab("How it works"):
        gr.Markdown(
            """
## The VERITAS pipeline

| Stage | Technique | Temperature |
|---|---|---|
| 1. Retrieve | Hybrid BM25 + TF-IDF cosine over sentence-window chunks | — |
| 2. Gate | Evidence-confidence threshold → abstain instead of guessing | — |
| 3. Generate | Citation-contract prompt: answer only from evidence, cite every sentence | 0.1 |
| 4. Decompose | Split the draft into atomic factual claims | 0.0 |
| 5. Verify | Two independent judges per claim: lexical entailment + LLM chain-of-verification | 0.0 |
| 6. Repair | Rewrite unsupported claims from evidence, or drop them; abstain if most of the draft fails | 0.0 |
| 7. Score | Groundedness = fraction of supported claims, shipped with the answer | — |

**Why it reduces hallucination**

- Most RAG hallucinations happen when the corpus can't answer the question — the gate abstains instead.
- The citation contract + low temperature suppress free-form confabulation during generation.
- Claim-level verification catches what still slips through (fabricated numbers are caught by the model-free lexical judge, which cannot be sweet-talked by the LLM).
- What can't be verified is repaired from evidence or removed — and the groundedness score makes any residual risk visible.

See `skills/hallucination-reduction/SKILL.md` in the repo for the reusable prompting / CoT / temperature playbook.
"""
        )

if __name__ == "__main__":
    demo.launch()
