"""The VERITAS pipeline and the vanilla-RAG baseline it is benchmarked against.

Stages of :class:`VeritasPipeline`:

1. retrieve (hybrid BM25 + TF-IDF) and gate on evidence confidence — abstain
   early when the corpus cannot support the question
2. generate a citation-constrained draft at low temperature
3. decompose the draft into atomic claims (temperature 0)
4. verify every claim with two independent judges (lexical entailment +
   LLM chain-of-verification at temperature 0)
5. repair or drop unsupported claims (one repair round, temperature 0)
6. assemble the final answer with a per-answer groundedness score; downgrade
   to an abstention when too little of the answer survives verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .claims import Claim, decompose_answer, parse_claim_line
from .llm import LLMClient
from .prompts import (
    ABSTAIN_TEXT,
    GROUNDED_SYSTEM,
    STAGE_TEMPERATURES,
    baseline_answer_prompt,
    grounded_answer_prompt,
    repair_prompt,
)
from .retrieval import HybridRetriever, ScoredChunk
from .verification import ClaimVerdict, Verdict, verify_claim


@dataclass
class PipelineConfig:
    top_k: int = 4
    confidence_threshold: float = 0.45   # abstain below this evidence confidence
    repair: bool = True                  # attempt to repair unsupported claims
    max_unsupported_fraction: float = 0.5  # abstain when > this fraction fails
    use_llm_verifier: bool = True        # lexical judge alone when False


@dataclass
class StageRecord:
    stage: str
    detail: str


@dataclass
class BaselineResult:
    question: str
    answer: str
    retrieved: List[ScoredChunk]


@dataclass
class VeritasResult:
    question: str
    answer: str
    abstained: bool
    abstain_reason: Optional[str]
    confidence: float
    retrieved: List[ScoredChunk] = field(default_factory=list)
    draft: str = ""
    draft_verdicts: List[ClaimVerdict] = field(default_factory=list)
    final_verdicts: List[ClaimVerdict] = field(default_factory=list)
    groundedness: float = 1.0
    repaired: int = 0
    removed: int = 0
    trace: List[StageRecord] = field(default_factory=list)

    @property
    def citations(self) -> List[str]:
        ids: List[str] = []
        for v in self.final_verdicts:
            for cid in v.claim.citations:
                if cid not in ids:
                    ids.append(cid)
        return ids


def groundedness_score(verdicts: Sequence[ClaimVerdict]) -> float:
    """Fraction of the answer that is evidence-supported (PARTIAL counts half)."""
    if not verdicts:
        return 1.0
    total = 0.0
    for v in verdicts:
        if v.label == Verdict.SUPPORTED:
            total += 1.0
        elif v.label == Verdict.PARTIAL:
            total += 0.5
    return total / len(verdicts)


class BaselineRAG:
    """Vanilla RAG: retrieve, stuff context, ask, trust the output."""

    def __init__(self, llm: LLMClient, retriever: HybridRetriever, top_k: int = 4):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, question: str) -> BaselineResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        raw = self.llm.complete(
            baseline_answer_prompt(question, retrieved),
            temperature=0.7,  # tutorial-default sampling, part of the baseline
            max_tokens=512,
        )
        return BaselineResult(question=question, answer=raw.strip(), retrieved=retrieved)


class VeritasPipeline:
    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        config: Optional[PipelineConfig] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ run
    def answer(self, question: str) -> VeritasResult:
        cfg = self.config
        trace: List[StageRecord] = []

        # 1. retrieve + confidence gate
        retrieved = self.retriever.retrieve(question, k=cfg.top_k)
        confidence = self.retriever.confidence(question, retrieved)
        trace.append(
            StageRecord(
                "retrieve",
                f"top-{len(retrieved)} chunks "
                f"{[sc.chunk.chunk_id for sc in retrieved]}, "
                f"evidence confidence {confidence:.2f} "
                f"(threshold {cfg.confidence_threshold:.2f})",
            )
        )
        if confidence < cfg.confidence_threshold:
            trace.append(StageRecord("gate", "confidence below threshold -> abstain"))
            return self._abstain(
                question, confidence, retrieved, trace, "low retrieval confidence"
            )

        # 2. grounded draft
        draft = self.llm.complete(
            grounded_answer_prompt(question, retrieved),
            system=GROUNDED_SYSTEM,
            temperature=STAGE_TEMPERATURES["generate"],
            max_tokens=512,
        ).strip()
        trace.append(StageRecord("generate", draft))
        if ABSTAIN_TEXT.lower() in draft.lower():
            trace.append(StageRecord("gate", "model reported insufficient evidence"))
            return self._abstain(
                question, confidence, retrieved, trace, "model abstained", draft=draft
            )

        # 3. claim decomposition
        claims = decompose_answer(draft, llm=self.llm)
        trace.append(
            StageRecord("decompose", f"{len(claims)} claims: " + " | ".join(c.text for c in claims))
        )

        # 4. verification
        verifier_llm = self.llm if cfg.use_llm_verifier else None
        draft_verdicts = [verify_claim(c, retrieved, llm=verifier_llm) for c in claims]
        trace.append(
            StageRecord(
                "verify",
                " | ".join(
                    f"{v.label.value} (lex {v.lexical_score:.2f}): {v.claim.text[:60]}"
                    for v in draft_verdicts
                ),
            )
        )

        # 5. repair / drop unsupported claims
        final_verdicts: List[ClaimVerdict] = []
        repaired = removed = 0
        for verdict in draft_verdicts:
            if verdict.label != Verdict.UNSUPPORTED:
                final_verdicts.append(verdict)
                continue
            fixed = self._try_repair(verdict.claim, retrieved, verifier_llm) if cfg.repair else None
            if fixed is not None:
                final_verdicts.append(fixed)
                repaired += 1
                trace.append(
                    StageRecord("repair", f"repaired -> {fixed.claim.raw}")
                )
            else:
                removed += 1
                trace.append(
                    StageRecord("repair", f"removed unsupported claim: {verdict.claim.text}")
                )

        # 6. assemble + final gate
        if not final_verdicts:
            trace.append(StageRecord("gate", "no claims survived verification -> abstain"))
            return self._abstain(
                question, confidence, retrieved, trace,
                "no verifiable claims", draft=draft, draft_verdicts=draft_verdicts,
            )
        unsupported_fraction = removed / len(draft_verdicts) if draft_verdicts else 0.0
        if unsupported_fraction > cfg.max_unsupported_fraction:
            trace.append(
                StageRecord(
                    "gate",
                    f"{unsupported_fraction:.0%} of draft claims unsupported -> abstain",
                )
            )
            return self._abstain(
                question, confidence, retrieved, trace,
                "answer mostly unsupported", draft=draft, draft_verdicts=draft_verdicts,
            )

        answer = " ".join(self._render_claim(v.claim) for v in final_verdicts)
        result = VeritasResult(
            question=question,
            answer=answer,
            abstained=False,
            abstain_reason=None,
            confidence=confidence,
            retrieved=retrieved,
            draft=draft,
            draft_verdicts=draft_verdicts,
            final_verdicts=final_verdicts,
            groundedness=groundedness_score(final_verdicts),
            repaired=repaired,
            removed=removed,
            trace=trace,
        )
        trace.append(
            StageRecord("score", f"groundedness {result.groundedness:.2f}, "
                                 f"{repaired} repaired, {removed} removed")
        )
        return result

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _render_claim(claim: Claim) -> str:
        text = claim.text if claim.text.endswith((".", "!", "?")) else claim.text + "."
        citations = "".join(f" [{c}]" for c in claim.citations)
        return f"{text}{citations}"

    def _try_repair(
        self,
        claim: Claim,
        retrieved: Sequence[ScoredChunk],
        verifier_llm: Optional[LLMClient],
    ) -> Optional[ClaimVerdict]:
        evidence_text = "\n".join(f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in retrieved)
        raw = self.llm.complete(
            repair_prompt(claim.text, evidence_text),
            temperature=STAGE_TEMPERATURES["repair"],
            max_tokens=256,
        ).strip()
        if not raw or raw.upper().startswith("REMOVE"):
            return None
        fixed = parse_claim_line(raw.splitlines()[0])
        if fixed is None:
            return None
        verdict = verify_claim(fixed, retrieved, llm=verifier_llm)
        if verdict.label == Verdict.UNSUPPORTED:
            return None
        return verdict

    def _abstain(
        self,
        question: str,
        confidence: float,
        retrieved: List[ScoredChunk],
        trace: List[StageRecord],
        reason: str,
        draft: str = "",
        draft_verdicts: Optional[List[ClaimVerdict]] = None,
    ) -> VeritasResult:
        return VeritasResult(
            question=question,
            answer=ABSTAIN_TEXT,
            abstained=True,
            abstain_reason=reason,
            confidence=confidence,
            retrieved=retrieved,
            draft=draft,
            draft_verdicts=draft_verdicts or [],
            final_verdicts=[],
            groundedness=1.0,  # an abstention asserts nothing unsupported
            trace=trace,
        )
