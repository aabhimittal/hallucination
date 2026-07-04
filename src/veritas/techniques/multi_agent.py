"""Multi-Agent Cross-Examination.

Three roles run in a loop: a **researcher** drafts a grounded answer, an
**editor** tightens it without adding facts, and a **judge** scores each claim's
alignment with the source using an NLI scorer. If the judge finds an
unsupported claim, it sends the draft back for one rewrite; claims that still
fail are dropped, and an answer that mostly fails is downgraded to abstention.

Black-box: the researcher/editor are ordinary LLM calls; the judge is a
pluggable NLI scorer (model-free by default).
"""

from __future__ import annotations

from typing import List

from ..claims import decompose_answer
from ..prompts import (
    ABSTAIN_TEXT,
    GROUNDED_SYSTEM,
    editor_prompt,
    grounded_answer_prompt,
)
from ..retrieval import HybridRetriever, ScoredChunk
from .base import BaseTechnique, TechniqueResult
from .nli import LexicalNLI, NLIScorer


class MultiAgentTechnique(BaseTechnique):
    name = "Multi-Agent Consensus"
    family = "verify"
    requires = "any"

    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        scorer: NLIScorer = None,
        entail_threshold: float = 0.5,
        top_k: int = 4,
        max_unsupported_fraction: float = 0.5,
    ):
        self.llm = llm
        self.retriever = retriever
        self.scorer = scorer or LexicalNLI()
        self.entail_threshold = entail_threshold
        self.top_k = top_k
        self.max_unsupported_fraction = max_unsupported_fraction

    def _judge(self, answer: str, retrieved: List[ScoredChunk]):
        """Return (kept_claims, n_failed) per NLI alignment with the source."""
        evidence_text = " ".join(sc.chunk.text for sc in retrieved)
        claims = decompose_answer(answer, llm=self.llm)
        kept, failed = [], 0
        for claim in claims:
            if self.scorer.entails(evidence_text, claim.text) >= self.entail_threshold:
                kept.append(claim)
            else:
                failed += 1
        return claims, kept, failed

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        trace = [f"retrieved {[sc.chunk.chunk_id for sc in retrieved]}"]

        # researcher
        draft = self.llm.complete(
            grounded_answer_prompt(question, retrieved),
            system=GROUNDED_SYSTEM, temperature=0.1,
        ).strip()
        trace.append(f"researcher draft: {draft}")
        if ABSTAIN_TEXT.lower() in draft.lower():
            return TechniqueResult(question, ABSTAIN_TEXT, True, groundedness=1.0,
                                   trace=trace + ["researcher found no evidence → abstain"])

        # editor
        edited = self.llm.complete(
            editor_prompt(question, draft, retrieved), temperature=0.0
        ).strip() or draft
        trace.append("editor pass complete")

        # judge, with one rewrite loop if contradictions found
        claims, kept, failed = self._judge(edited, retrieved)
        trace.append(f"judge: {len(claims)} claims, {failed} failed NLI alignment")
        if failed and kept:
            # send the surviving claims back as a corrected draft (one loop)
            edited = " ".join(c.raw or c.text for c in kept)
            claims, kept, failed = self._judge(edited, retrieved)
            trace.append(f"rewrite: {len(kept)} claims kept after re-judging")

        if not kept or (claims and failed / len(claims) > self.max_unsupported_fraction):
            return TechniqueResult(question, ABSTAIN_TEXT, True, groundedness=1.0,
                                   trace=trace + ["consensus not reached → abstain"])

        answer = " ".join(c.raw or c.text for c in kept)
        return TechniqueResult(
            question=question,
            answer=answer,
            abstained=False,
            groundedness=len(kept) / len(claims) if claims else 1.0,
            trace=trace + ["consensus reached"],
            extra={"claims": len(claims), "kept": len(kept)},
        )
