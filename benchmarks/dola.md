# DoLa (white-box) comparison

- Model: **gpt2** (local), dola_layers=`high`
- Questions: 8
- Vanilla greedy decoding vs DoLa (contrasting layers) on the same model.

| Metric | Vanilla decoding | DoLa |
|---|---|---|
| Hallucination rate ↓ | 100.0% | 100.0% |
| Answer accuracy ↑ | 0.0% | 12.5% |
| Mean groundedness ↑ | 2.1% | 12.1% |
| Mean latency (s) ↓ | 1.82 | 3.64 |

Model `gpt2` is a small base model — see the script docstring on interpreting small-model numbers.
