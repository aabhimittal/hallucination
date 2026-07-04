#!/usr/bin/env python3
"""Quantitatively compare every hallucination-reduction technique.

Runs all black-box techniques (+ Graph-RAG) over the bundled dataset on the same
deterministic MockLLM and grades them with the same independent judge, so the
numbers are directly comparable. Writes ``benchmarks/comparison.md`` and
``benchmarks/comparison.json``.

    python benchmarks/run_comparison.py
    python benchmarks/run_comparison.py --hallucination-rate 0.5 --seed 7
    python benchmarks/run_comparison.py --provider anthropic     # live model

DoLa (white-box) is compared separately via ``run_dola.py`` because it needs a
local Hugging Face model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from veritas import HybridRetriever, MockLLM, chunk_corpus
from veritas.chunking import load_documents_from_dir
from veritas.graph import GraphRetriever
from veritas.metrics import (
    QuestionRecord,
    calibration_report,
    evaluate_run,
)
from veritas.techniques import build_registry
from veritas.techniques.graph_rag import GraphRAGTechnique

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
    if args.provider == "local":
        from veritas.local import LocalChatClient
        return LocalChatClient(model=args.model or "Qwen/Qwen2.5-7B-Instruct",
                               load_4bit=args.load_4bit)
    raise ValueError(args.provider)


def run_technique(technique, questions):
    records = []
    for item in questions:
        start = time.perf_counter()
        res = technique.answer(item["question"])
        latency = time.perf_counter() - start
        records.append(
            QuestionRecord(
                question=item["question"],
                qtype=item["type"],
                gold_keywords=item["gold_keywords"],
                answer=res.answer,
                abstained=res.abstained,
                latency_s=latency,
                confidence=res.confidence,
            )
        )
    return records


def fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="mock",
                   choices=["mock", "anthropic", "openai", "hf", "local"])
    p.add_argument("--model", default=None)
    p.add_argument("--load-4bit", action="store_true",
                   help="4-bit quantize a local model (fits a 7-8B model on a free T4 GPU)")
    p.add_argument("--hallucination-rate", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N questions (for quick/live runs)")
    p.add_argument("--balanced", type=int, default=None,
                   help="run first N of each question type (answerable/unanswerable/adversarial)")
    p.add_argument("--out", default="comparison",
                   help="output basename under benchmarks/ (e.g. 'comparison_live')")
    args = p.parse_args()

    docs = load_documents_from_dir(HERE / "corpus")
    chunks = chunk_corpus(docs)
    retriever = HybridRetriever(chunks)
    graph_retriever = GraphRetriever(chunks)
    dataset = json.loads((HERE / "dataset.json").read_text())
    questions = dataset["questions"]
    if args.balanced:
        # first N of each type — keeps a live run fast while covering all metrics
        picked = []
        for qtype in ("answerable", "unanswerable", "adversarial"):
            picked += [q for q in questions if q["type"] == qtype][: args.balanced]
        questions = picked
    elif args.limit:
        questions = questions[: args.limit]

    llm = build_llm(args)
    techniques = build_registry(llm, retriever)
    techniques[GraphRAGTechnique.name] = GraphRAGTechnique(llm, graph_retriever)

    print(f"Comparing {len(techniques)} techniques over {len(questions)} questions "
          f"(provider={args.provider})")

    results = {}
    for name, technique in techniques.items():
        records = run_technique(technique, questions)
        metrics = evaluate_run(records, chunks)
        metrics.update({f"cal_{k}": v
                        for k, v in calibration_report(records, chunks).items()})
        results[name] = {"family": technique.family, "metrics": metrics}
        print(f"  {name:34s} hallucination {fmt(metrics['hallucination_rate'])}, "
              f"groundedness {fmt(metrics['mean_groundedness'])}, "
              f"abstention-recall {fmt(metrics['abstention_recall'])}")

    # --------------------------------------------------------------- report
    rows = [
        ("Hallucination rate ↓", "hallucination_rate", True),
        ("Unsupported claim rate ↓", "unsupported_claim_rate", True),
        ("Mean groundedness ↑", "mean_groundedness", True),
        ("Abstention recall ↑", "abstention_recall", True),
        ("False abstention rate ↓", "false_abstention_rate", True),
        ("Answer accuracy ↑", "answer_accuracy", True),
        ("Answer coverage ↑", "answer_coverage", True),
        ("Calibration ECE ↓", "cal_ece", False),
        ("Calibration AUROC ↑", "cal_auroc", False),
        ("Risk–coverage AURC ↓", "cal_aurc", False),
        ("Mean latency (s) ↓", "mean_latency_s", False),
    ]
    names = list(results)
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|---|" + "|".join(["---"] * len(names)) + "|"
    lines = [
        "# Technique comparison",
        "",
        f"- Questions: **{len(questions)}** "
        f"({sum(1 for q in questions if q['type'] == 'answerable')} answerable, "
        f"{sum(1 for q in questions if q['type'] == 'unanswerable')} unanswerable, "
        f"{sum(1 for q in questions if q['type'] == 'adversarial')} adversarial)",
        f"- Provider: **{args.provider}**"
        + (f" (MockLLM, {args.hallucination_rate:.0%} injected hallucinations, seed {args.seed})"
           if args.provider == "mock" else f" (model {args.model})"),
        "- Same corpus, same independent lexical judge for every technique. "
        "↑ = higher is better, ↓ = lower is better.",
        "",
        header, sep,
    ]
    for label, key, pct in rows:
        cells = [fmt(results[n]["metrics"].get(key), pct) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "**Families:** " + ", ".join(
            sorted(set(f"{n} ({results[n]['family']})" for n in names))
        ),
        "",
        "Notes: Semantic Entropy and the abstaining techniques trade coverage "
        "for safety — read hallucination rate together with false-abstention "
        "rate. Graph-RAG matches flat retrieval on single-hop factoids; its "
        "multi-hop advantage needs a reasoning LLM and relational corpus (the "
        "offline mock answers by lexical overlap). DoLa (white-box) is compared "
        "separately — see `benchmarks/dola.md` / `run_dola.py`.",
        "",
        "Reproduce: `python benchmarks/run_comparison.py`",
        "",
    ]
    md = "\n".join(lines)
    (HERE / f"{args.out}.md").write_text(md)
    (HERE / f"{args.out}.json").write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2))
    print("\n" + md)
    print(f"Wrote {HERE / (args.out + '.md')} and {HERE / (args.out + '.json')}")


if __name__ == "__main__":
    main()
