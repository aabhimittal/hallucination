"""Per-claim verification: lexical entailment + LLM chain-of-verification.

Two independent judges look at every claim:

1. :func:`lexical_entailment` — a model-free scorer based on content-token
   coverage and strict number agreement. It cannot be fooled by a persuasive
   LLM, which makes it a strong tripwire for fabricated numbers/entities.
2. An LLM verifier running the chain-of-verification (CoVe) prompt at
   temperature 0.

The two verdicts are fused conservatively: a claim is only SUPPORTED when both
judges agree, and any disagreement demotes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from .claims import Claim
from .llm import LLMClient
from .prompts import STAGE_TEMPERATURES, verify_prompt
from .retrieval import ScoredChunk, content_tokens, tokenize

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_VERDICT_RE = re.compile(r"VERDICT:\s*(SUPPORTED|PARTIAL|UNSUPPORTED)")


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


def _numbers(text: str) -> List[str]:
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text)]


def lexical_entailment(claim: str, evidence_text: str) -> float:
    """Score in [0, 1]: how well the evidence lexically entails the claim."""
    claim_terms = content_tokens(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = set(tokenize(evidence_text))
    coverage = sum(1 for t in claim_terms if t in evidence_terms) / len(claim_terms)

    claim_numbers = _numbers(claim)
    if claim_numbers:
        evidence_numbers = set(_numbers(evidence_text))
        if not all(n in evidence_numbers for n in claim_numbers):
            # a number the evidence doesn't contain is the classic fabrication
            coverage = min(coverage, 0.3)
    return coverage


@dataclass
class ClaimVerdict:
    claim: Claim
    label: Verdict
    lexical_score: float
    llm_verdict: Optional[Verdict] = None
    evidence_ids: List[str] = field(default_factory=list)


def parse_llm_verdict(raw: str) -> Verdict:
    match = _VERDICT_RE.search(raw)
    if not match:
        # An unparseable verifier response must never count as support.
        return Verdict.UNSUPPORTED
    return Verdict(match.group(1))


def _evidence_for_claim(
    claim: Claim, retrieved: Sequence[ScoredChunk]
) -> Sequence[ScoredChunk]:
    """Prefer the chunks the claim cites; fall back to all retrieved chunks."""
    if claim.citations:
        cited = [sc for sc in retrieved if sc.chunk.chunk_id in claim.citations]
        if cited:
            return cited
    return retrieved


def verify_claim(
    claim: Claim,
    retrieved: Sequence[ScoredChunk],
    llm: Optional[LLMClient] = None,
    lex_support: float = 0.65,
    lex_reject: float = 0.4,
) -> ClaimVerdict:
    """Verify one claim against evidence; fuse lexical + LLM verdicts."""
    evidence = _evidence_for_claim(claim, retrieved)
    evidence_text = "\n".join(
        f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in evidence
    )
    lex = lexical_entailment(claim.text, evidence_text)
    lex_verdict = (
        Verdict.SUPPORTED
        if lex >= lex_support
        else Verdict.UNSUPPORTED if lex < lex_reject else Verdict.PARTIAL
    )

    llm_verdict: Optional[Verdict] = None
    if llm is not None:
        raw = llm.complete(
            verify_prompt(claim.text, evidence_text),
            temperature=STAGE_TEMPERATURES["verify"],
            max_tokens=400,
        )
        llm_verdict = parse_llm_verdict(raw)

    label = _fuse(lex_verdict, llm_verdict)
    return ClaimVerdict(
        claim=claim,
        label=label,
        lexical_score=lex,
        llm_verdict=llm_verdict,
        evidence_ids=[sc.chunk.chunk_id for sc in evidence],
    )


def _fuse(lex: Verdict, llm: Optional[Verdict]) -> Verdict:
    """Conservative fusion: SUPPORTED needs agreement; disagreement demotes."""
    if llm is None:
        return lex
    if lex == llm:
        return lex
    if Verdict.UNSUPPORTED in (lex, llm):
        # one judge rejecting outright caps the claim at PARTIAL, and if the
        # other judge is only PARTIAL the claim falls to UNSUPPORTED
        return Verdict.UNSUPPORTED if Verdict.PARTIAL in (lex, llm) else Verdict.PARTIAL
    # SUPPORTED vs PARTIAL
    return Verdict.PARTIAL


def verify_answer(
    claims: Sequence[Claim],
    retrieved: Sequence[ScoredChunk],
    llm: Optional[LLMClient] = None,
) -> List[ClaimVerdict]:
    return [verify_claim(c, retrieved, llm=llm) for c in claims]
