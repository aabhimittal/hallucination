"""Claim decomposition: split an answer into atomic, citable factual claims."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .chunking import split_cited_sentences
from .llm import LLMClient
from .prompts import STAGE_TEMPERATURES, decompose_prompt

_CITATION_RE = re.compile(r"\[(c\d+)\]")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$")


@dataclass
class Claim:
    text: str                       # claim text with citation markers stripped
    citations: List[str] = field(default_factory=list)
    raw: str = ""                   # original line, citations included


def parse_claim_line(line: str) -> Optional[Claim]:
    match = _NUMBERED_RE.match(line)
    body = match.group(1).strip() if match else line.strip()
    if not body:
        return None
    citations = _CITATION_RE.findall(body)
    text = _CITATION_RE.sub("", body)
    text = " ".join(text.split()).strip()
    if not text:
        return None
    return Claim(text=text, citations=citations, raw=body)


def split_into_claims_fallback(answer: str) -> List[Claim]:
    """Deterministic fallback: one claim per sentence."""
    sentences = split_cited_sentences(answer)
    claims = []
    for sent in sentences:
        claim = parse_claim_line(sent)
        if claim:
            claims.append(claim)
    return claims


def decompose_answer(answer: str, llm: Optional[LLMClient] = None) -> List[Claim]:
    """Decompose ``answer`` into atomic claims.

    Uses the LLM at temperature 0 when available; falls back to sentence
    splitting when the LLM output is unusable or no LLM is given.
    """
    if llm is not None:
        raw = llm.complete(
            decompose_prompt(answer),
            temperature=STAGE_TEMPERATURES["decompose"],
            max_tokens=512,
        )
        claims = []
        for line in raw.splitlines():
            claim = parse_claim_line(line)
            if claim:
                claims.append(claim)
        if claims:
            return claims
    return split_into_claims_fallback(answer)
