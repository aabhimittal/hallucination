#!/usr/bin/env python3
"""Benchmark BaselineRAG vs the VERITAS pipeline on the bundled dataset.

Default run is fully offline and reproducible: both systems sit on top of the
same deterministic ``MockLLM`` configured to hallucinate on a fraction of its
answers (simulating an unreliable base model). The benchmark then measures how
much of that noise each system lets through.

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --hallucination-rate 0.5 --seed 7
    python benchmarks/run_benchmark.py --provider anthropic --model claude-opus-4-8

Writes ``benchmarks/results.json`` and ``benchmarks/results.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from veritas import (
    BaselineRAG,
    HybridRetriever,
    MockLLM,
    VeritasPipeline,
    chunk_corpus,
)
from veritas.chunking import load_documents_from_dir
from veritas.metrics import (
    QuestionRecord,
    citation_precision,
    evaluate_run,
    is_abstention,
)

HERE = Path(__file__).parent


def build_llm(args):
    if args.provider == "mock":
        return MockLLM(hallucination_rate=args.hallucination_rate, seed=args.seed)
    if args.provider == "anthropic":
        from veritas import AnthropicClient

        return AnthropicClient(model=args.model or "claude-opus-4-8")
    if args.provider == "openai":
        from veritas import OpenAICompatClient

        return OpenAICompatClient(model=args.model or "gpt-4o-mini")
    if args.provider == "hf":
        from veritas import HFInferenceClient

        return HFInferenceClient(model=args.model or "meta-llama/Llama-3.1-8B-Instruct")
    raise ValueError(f"unknown provider {args.provider}")


def run_system(name, answer_fn, questions):
    """Run one system over all questions; returns (records, extra)."""
    records = []
    verdicts = []
    for item in questions:
        start = time.perf_counter()
        result = answer_fn(item["question"])
        latency = time.perf_counter() - start
        if hasattr(result, "abstained"):  # VeritasResult
            answer = result.answer
            abstained = result.abstained
            verdicts.extend(result.final_verdicts)
        else:  # BaselineResult
            answer = result.answer
            abstained = is_abstention(answer)
        records.append(
            QuestionRecord(
                question=item["question"],
                qtype=item["type"],
                gold_keywords=item["gold_keywords"],
                answer=answer,
                abstained=abstained,
                latency_s=latency,
            )
        )
    return records, verdicts


def fmt(value, pct=True):
    if value is None:
        return "—"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "hf"])
    parser.add_argument("--model", default=None, help="model id for the chosen provider")
    parser.add_argument("--hallucination-rate", type=float, default=0.35,
                        help="MockLLM hallucination injection rate (mock provider only)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    docs = load_documents_from_dir(HERE / "corpus")
    chunks = chunk_corpus(docs)
    retriever = HybridRetriever(chunks)
    dataset = json.loads((HERE / "dataset.json").read_text())
    questions = dataset["questions"]

    llm = build_llm(args)
    baseline = BaselineRAG(llm, retriever, top_k=args.top_k)
    veritas = VeritasPipeline(llm, retriever)

    print(f"Running benchmark: {len(questions)} questions, provider={args.provider}, "
          f"hallucination_rate={args.hallucination_rate if args.provider == 'mock' else 'n/a'}")

    base_records, _ = run_system("baseline", baseline.answer, questions)
    base_metrics = evaluate_run(base_records, chunks)

    ver_records, ver_verdicts = run_system("veritas", veritas.answer, questions)
    ver_metrics = evaluate_run(ver_records, chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    ver_metrics["citation_precision"] = citation_precision(ver_verdicts, chunks_by_id)
    base_metrics["citation_precision"] = None  # baseline emits no citations

    rows = [
        ("Hallucination rate (per question)", "hallucination_rate", True, "lower"),
        ("Unsupported claim rate", "unsupported_claim_rate", True, "lower"),
        ("Mean groundedness", "mean_groundedness", True, "higher"),
        ("Abstention recall (unanswerable)", "abstention_recall", True, "higher"),
        ("False abstention rate (answerable)", "false_abstention_rate", True, "lower"),
        ("Answer accuracy (answerable)", "answer_accuracy", True, "higher"),
        ("Answer coverage (answerable)", "answer_coverage", True, "higher"),
        ("Citation precision", "citation_precision", True, "higher"),
        ("Mean latency (s)", "mean_latency_s", False, "lower"),
    ]

    md = [
        "# Benchmark results: Baseline RAG vs VERITAS",
        "",
        f"- Questions: **{len(questions)}** "
        f"({sum(1 for q in questions if q['type'] == 'answerable')} answerable, "
        f"{sum(1 for q in questions if q['type'] == 'unanswerable')} unanswerable, "
        f"{sum(1 for q in questions if q['type'] == 'adversarial')} adversarial)",
        f"- Provider: **{args.provider}**"
        + (f" (MockLLM, hallucination injection rate {args.hallucination_rate:.0%}, seed {args.seed})"
           if args.provider == "mock" else f" (model {args.model})"),
        "- Judge: model-free lexical-entailment grader against the full corpus "
        "(same ruler for both systems). Answering an unanswerable/adversarial "
        "question at all counts as a hallucinated question.",
        "",
        "| Metric | Baseline RAG | VERITAS | Better |",
        "|---|---|---|---|",
    ]
    for label, key, pct, better in rows:
        md.append(
            f"| {label} | {fmt(base_metrics.get(key), pct)} | "
            f"{fmt(ver_metrics.get(key), pct)} | {better} |"
        )
    md += [
        "",
        "Reproduce with: `python benchmarks/run_benchmark.py`",
        "",
    ]
    md_text = "\n".join(md)
    (HERE / "results.md").write_text(md_text)
    (HERE / "results.json").write_text(json.dumps(
        {
            "config": vars(args),
            "baseline": base_metrics,
            "veritas": ver_metrics,
            "answers": {
                "baseline": [
                    {"q": r.question, "type": r.qtype, "answer": r.answer,
                     "abstained": r.abstained} for r in base_records
                ],
                "veritas": [
                    {"q": r.question, "type": r.qtype, "answer": r.answer,
                     "abstained": r.abstained} for r in ver_records
                ],
            },
        },
        indent=2,
    ))
    print(md_text)
    print(f"Wrote {HERE / 'results.md'} and {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
