#!/usr/bin/env python3
"""White-box decoding comparison: vanilla greedy vs DoLa, on a local HF model.

DoLa needs per-layer logits, so this runs only where ``torch`` + ``transformers``
are installed and a model can be loaded locally (the "local-hf" runtime). It is
kept separate from ``run_comparison.py`` (which is model-agnostic and offline)
for exactly that reason.

    pip install 'veritas-rag[local]'
    python benchmarks/run_dola.py --model gpt2 --dola-layers high

Note: DoLa's factuality gains are strongest on larger *instruction-tuned* models
evaluated on a factuality benchmark (e.g. TruthfulQA). A small base model like
gpt2 on CPU is enough to exercise the mechanism end-to-end but is not expected
to show DoLa's full effect — treat small-model numbers as a wiring check, not a
verdict on DoLa.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from veritas import HybridRetriever, chunk_corpus
from veritas.chunking import load_documents_from_dir
from veritas.metrics import QuestionRecord, evaluate_run

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dola-layers", default="high", help="'high' | 'low' | comma list")
    p.add_argument("--limit", type=int, default=12, help="questions to run (CPU-bound)")
    args = p.parse_args()

    try:
        from veritas.techniques.decoding import build_decoding_registry
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"local-hf runtime unavailable ({exc}). "
            f"Install with: pip install 'veritas-rag[local]'"
        )

    dola_layers = args.dola_layers
    if "," in dola_layers:
        dola_layers = [int(x) for x in dola_layers.split(",")]

    docs = load_documents_from_dir(HERE / "corpus")
    chunks = chunk_corpus(docs)
    retriever = HybridRetriever(chunks)
    questions = json.loads((HERE / "dataset.json").read_text())["questions"][: args.limit]

    print(f"Loading {args.model} (this downloads on first run)...")
    techniques = build_decoding_registry(retriever, model=args.model,
                                          dola_layers=dola_layers)

    results = {}
    for name, tech in techniques.items():
        records = []
        for item in questions:
            t0 = time.perf_counter()
            res = tech.answer(item["question"])
            records.append(QuestionRecord(
                item["question"], item["type"], item["gold_keywords"],
                res.answer, res.abstained, time.perf_counter() - t0))
        results[name] = evaluate_run(records, chunks)
        print(f"  {name}: hallucination {results[name]['hallucination_rate']:.0%}, "
              f"accuracy {results[name]['answer_accuracy']}")

    lines = [
        "# DoLa (white-box) comparison",
        "",
        f"- Model: **{args.model}** (local), dola_layers=`{args.dola_layers}`",
        f"- Questions: {len(questions)}",
        "- Vanilla greedy decoding vs DoLa (contrasting layers) on the same model.",
        "",
        "| Metric | Vanilla decoding | DoLa |",
        "|---|---|---|",
    ]
    def fmt(x, pct=True):
        if x is None:
            return "—"
        return f"{x:.1%}" if pct else f"{x:.2f}"

    for label, key, pct in [
        ("Hallucination rate ↓", "hallucination_rate", True),
        ("Answer accuracy ↑", "answer_accuracy", True),
        ("Mean groundedness ↑", "mean_groundedness", True),
        ("Mean latency (s) ↓", "mean_latency_s", False),
    ]:
        v = fmt(results["Vanilla decoding (local)"].get(key), pct)
        d = fmt(results["DoLa (local)"].get(key), pct)
        lines.append(f"| {label} | {v} | {d} |")
    lines += ["", f"Model `{args.model}` is a small base model — see the script "
              "docstring on interpreting small-model numbers.", ""]
    md = "\n".join(lines)
    (HERE / "dola.md").write_text(md)
    (HERE / "dola.json").write_text(json.dumps(results, indent=2))
    print("\n" + md)


if __name__ == "__main__":
    main()
