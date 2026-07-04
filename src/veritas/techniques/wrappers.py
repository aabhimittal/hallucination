"""Adapt the existing BaselineRAG and VeritasPipeline to the Technique API."""

from __future__ import annotations

from ..metrics import is_abstention
from ..pipeline import BaselineRAG, VeritasPipeline
from .base import BaseTechnique, TechniqueResult


class BaselineTechnique(BaseTechnique):
    name = "Baseline RAG"
    family = "baseline"
    requires = "any"

    def __init__(self, llm, retriever, top_k: int = 4):
        self._rag = BaselineRAG(llm, retriever, top_k=top_k)

    def answer(self, question: str) -> TechniqueResult:
        result = self._rag.answer(question)
        return TechniqueResult(
            question=question,
            answer=result.answer,
            abstained=is_abstention(result.answer),
            groundedness=None,
            trace=[f"retrieved {[sc.chunk.chunk_id for sc in result.retrieved]}",
                   "answered at temperature 0.7, no verification"],
        )


class VeritasTechnique(BaseTechnique):
    name = "VERITAS"
    family = "verify"
    requires = "any"

    def __init__(self, llm, retriever, config=None):
        self._pipeline = VeritasPipeline(llm, retriever, config)

    def answer(self, question: str) -> TechniqueResult:
        r = self._pipeline.answer(question)
        return TechniqueResult(
            question=question,
            answer=r.answer,
            abstained=r.abstained,
            confidence=r.confidence,
            groundedness=r.groundedness,
            trace=[f"{s.stage}: {s.detail}" for s in r.trace],
            extra={"repaired": r.repaired, "removed": r.removed,
                   "citations": r.citations},
        )
