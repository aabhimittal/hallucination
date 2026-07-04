"""Calibration + Selective Prediction.

The deployable, benchmarkable cousin of "Calibrated Uncertainty RLHF" (which is
a *training* method we can't run at inference). The model emits an answer plus a
self-reported confidence; the technique optionally abstains below a confidence
threshold (selective prediction). Across the dataset we then measure how well
those confidences are *calibrated* (see ``veritas.metrics``: ECE, AUROC,
risk-coverage) — i.e. whether the model's stated doubt actually tracks its
correctness.

Black-box: works on any LLM that can follow the "state your confidence" prompt.
"""

from __future__ import annotations

import re
from typing import Optional

from ..prompts import ABSTAIN_TEXT, GROUNDED_SYSTEM, answer_with_confidence_prompt
from ..retrieval import HybridRetriever
from .base import BaseTechnique, TechniqueResult

_CONF_RE = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.IGNORECASE)


def parse_confidence(raw: str) -> Optional[float]:
    match = _CONF_RE.search(raw)
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return None


class CalibrationTechnique(BaseTechnique):
    name = "Calibrated Selective Prediction"
    family = "uncertainty"
    requires = "any"

    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        confidence_threshold: float = 0.5,
        top_k: int = 4,
    ):
        self.llm = llm
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        raw = self.llm.complete(
            answer_with_confidence_prompt(question, retrieved),
            system=GROUNDED_SYSTEM, temperature=0.0,
        ).strip()
        confidence = parse_confidence(raw)
        answer = _CONF_RE.sub("", raw).strip()
        model_abstained = ABSTAIN_TEXT.lower() in answer.lower()
        if model_abstained:
            answer = ABSTAIN_TEXT

        trace = [f"stated confidence: {confidence}"]
        # selective prediction: withhold low-confidence answers
        if not model_abstained and confidence is not None \
                and confidence < self.confidence_threshold:
            trace.append(
                f"confidence {confidence:.2f} < {self.confidence_threshold} "
                f"→ withheld (selective prediction)"
            )
            return TechniqueResult(question, ABSTAIN_TEXT, True, confidence=confidence,
                                   trace=trace)

        return TechniqueResult(
            question=question,
            answer=answer,
            abstained=model_abstained,
            confidence=confidence,
            trace=trace,
        )
