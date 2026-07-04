"""Neurosymbolic Guardrails.

Rigid, code-based rails wrapped around a loose generative model (the idea
behind NeMo-Guardrails), implemented in pure Python:

- **input rail**: is the question in scope for the corpus? (reuses the
  retrieval-confidence signal) — out-of-scope questions are refused before
  generation.
- **output rails**: the generated answer must satisfy programmatic constraints
  — every sentence carries a citation, length is bounded, and no hedge/­
  speculation patterns ("I think", "probably", "as far as I know") slip
  through uncited. Any violation → the answer is rejected and the system
  abstains rather than emit an unconstrained response.

Deterministic and model-agnostic.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..chunking import split_cited_sentences
from ..prompts import ABSTAIN_TEXT, GROUNDED_SYSTEM, grounded_answer_prompt
from ..retrieval import HybridRetriever
from .base import BaseTechnique, TechniqueResult

_CIT_RE = re.compile(r"\[c\d+\]")
_SPECULATION = re.compile(
    r"\b(i think|i believe|probably|might be|as far as i know|i guess|"
    r"presumably|it seems|i'm not sure|maybe)\b",
    re.IGNORECASE,
)


def check_output_rails(answer: str, max_sentences: int = 6) -> Tuple[bool, List[str]]:
    """Return (passed, violations)."""
    violations: List[str] = []
    sentences = split_cited_sentences(answer)
    if len(sentences) > max_sentences:
        violations.append(f"answer exceeds {max_sentences} sentences")
    if _SPECULATION.search(answer):
        violations.append("contains speculation/hedging language")
    for sent in sentences:
        if not _CIT_RE.search(sent):
            violations.append("a sentence lacks a citation")
            break
    return (not violations), violations


class GuardrailsTechnique(BaseTechnique):
    name = "Neurosymbolic Guardrails"
    family = "guardrail"
    requires = "any"

    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        scope_threshold: float = 0.45,
        top_k: int = 4,
        max_sentences: int = 6,
    ):
        self.llm = llm
        self.retriever = retriever
        self.scope_threshold = scope_threshold
        self.top_k = top_k
        self.max_sentences = max_sentences

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        confidence = self.retriever.confidence(question, retrieved)
        trace = [f"input rail: scope confidence {confidence:.2f} "
                 f"(threshold {self.scope_threshold})"]

        # input rail
        if confidence < self.scope_threshold:
            return TechniqueResult(
                question, ABSTAIN_TEXT, True, confidence=confidence, groundedness=1.0,
                trace=trace + ["question out of corpus scope → refused by input rail"],
            )

        answer = self.llm.complete(
            grounded_answer_prompt(question, retrieved),
            system=GROUNDED_SYSTEM, temperature=0.1,
        ).strip()
        if ABSTAIN_TEXT.lower() in answer.lower():
            return TechniqueResult(question, ABSTAIN_TEXT, True, confidence=confidence,
                                   groundedness=1.0, trace=trace + ["model abstained"])

        # output rails
        passed, violations = check_output_rails(answer, self.max_sentences)
        if not passed:
            return TechniqueResult(
                question, ABSTAIN_TEXT, True, confidence=confidence, groundedness=1.0,
                trace=trace + [f"output rail blocked answer: {'; '.join(violations)}"],
                extra={"violations": violations},
            )
        return TechniqueResult(
            question=question, answer=answer, abstained=False, confidence=confidence,
            trace=trace + ["output rails passed"],
        )
