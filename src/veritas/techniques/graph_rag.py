"""Graph-RAG technique: grounded answering over a multi-hop knowledge graph.

Retrieval is done by :class:`veritas.graph.GraphRetriever` (entity graph +
multi-hop traversal) instead of flat lexical search, with a graph-grounded
confidence gate. Everything downstream (citation contract, abstention) mirrors
the other grounded techniques so the comparison isolates the *retrieval*
contribution.
"""

from __future__ import annotations

from ..graph import GraphRetriever
from ..prompts import ABSTAIN_TEXT, GROUNDED_SYSTEM, grounded_answer_prompt
from .base import BaseTechnique, TechniqueResult


class GraphRAGTechnique(BaseTechnique):
    name = "Graph-RAG"
    family = "graph"
    requires = "any"

    def __init__(
        self,
        llm,
        graph_retriever: GraphRetriever,
        confidence_threshold: float = 0.4,
        top_k: int = 4,
    ):
        self.llm = llm
        self.retriever = graph_retriever
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        confidence = self.retriever.confidence(question, retrieved)
        seeds = self.retriever.graph.match_entities(question)
        trace = [
            f"seed entities: {sorted(seeds) or '—'}",
            f"multi-hop chunks: {[sc.chunk.chunk_id for sc in retrieved]}, "
            f"graph confidence {confidence:.2f} (threshold {self.confidence_threshold})",
        ]
        if confidence < self.confidence_threshold or not retrieved:
            return TechniqueResult(
                question, ABSTAIN_TEXT, True, confidence=confidence, groundedness=1.0,
                trace=trace + ["no supporting subgraph → abstain"],
            )
        answer = self.llm.complete(
            grounded_answer_prompt(question, retrieved),
            system=GROUNDED_SYSTEM, temperature=0.1,
        ).strip()
        abstained = ABSTAIN_TEXT.lower() in answer.lower()
        return TechniqueResult(
            question=question,
            answer=answer if not abstained else ABSTAIN_TEXT,
            abstained=abstained,
            confidence=confidence,
            trace=trace + ["answered from multi-hop subgraph"],
            extra={"seed_entities": sorted(seeds)},
        )
