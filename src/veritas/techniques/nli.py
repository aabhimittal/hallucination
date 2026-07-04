"""Pluggable NLI (natural language inference) scorer for entailment checks.

Used by the multi-agent judge and semantic-entropy clustering. Defaults to the
model-free lexical-entailment scorer so everything runs offline; if a Hugging
Face NLI model is available it can be swapped in via ``HFNLIScorer`` without
touching callers.
"""

from __future__ import annotations

from typing import Protocol

from ..verification import lexical_entailment


class NLIScorer(Protocol):
    def entails(self, premise: str, hypothesis: str) -> float:
        """P(premise entails hypothesis) in [0, 1]."""
        ...


class LexicalNLI:
    """Model-free entailment proxy: content-token coverage + number matching."""

    name = "lexical"

    def entails(self, premise: str, hypothesis: str) -> float:
        return lexical_entailment(hypothesis, premise)


class HFNLIScorer:  # pragma: no cover - exercised only with transformers + a model
    """Real NLI via a Hugging Face sequence-classification model (e.g. DeBERTa-MNLI)."""

    name = "hf-nli"

    def __init__(self, model: str = "microsoft/deberta-large-mnli", pipeline=None):
        if pipeline is not None:
            self._pipe = pipeline
        else:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline("text-classification", model=model, top_k=None)

    def entails(self, premise: str, hypothesis: str) -> float:
        scores = self._pipe({"text": premise, "text_pair": hypothesis})
        for entry in scores:
            if entry["label"].lower().startswith("entail"):
                return float(entry["score"])
        return 0.0


def bidirectional_equivalent(a: str, b: str, scorer: NLIScorer, threshold: float = 0.6) -> bool:
    """Two texts share meaning if each entails the other (Kuhn et al. 2023)."""
    return scorer.entails(a, b) >= threshold and scorer.entails(b, a) >= threshold
