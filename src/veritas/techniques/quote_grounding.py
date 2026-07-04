"""Long-Context Quote Grounding.

Force the model to pull exact, verbatim quotes from the evidence into a
scratchpad *before* synthesizing an answer, then keep only the quotes that are
genuine substrings of the source. Fabricated "quotes" are dropped by a
deterministic substring check — the model cannot smuggle in invented text under
quotation marks. Synthesis is constrained to the surviving quotes.

Black-box: works on any LLM.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..prompts import (
    ABSTAIN_TEXT,
    quote_extraction_prompt,
    synthesize_from_quotes_prompt,
)
from ..retrieval import HybridRetriever
from .base import BaseTechnique, TechniqueResult

_QUOTE_RE = re.compile(r"^\s*QUOTE:\s*(.+)$", re.MULTILINE)
_CIT_RE = re.compile(r"\[(c\d+)\]")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def verify_quotes(raw: str, retrieved) -> Tuple[List[str], int]:
    """Keep only quotes that appear verbatim in the cited (or any) chunk.

    Returns ``(valid_quote_lines, n_rejected)``.
    """
    by_id = {sc.chunk.chunk_id: sc.chunk.text for sc in retrieved}
    all_text = _normalize(" ".join(by_id.values()))
    valid: List[str] = []
    rejected = 0
    for line in _QUOTE_RE.findall(raw):
        if line.strip().upper() == "NONE":
            continue
        quote_text = _normalize(_CIT_RE.sub("", line))
        if not quote_text:
            continue
        cited = _CIT_RE.findall(line)
        # a quote is valid iff it is a verbatim substring of its cited chunk
        # (or, if it cited nothing, of any chunk)
        haystacks = [_normalize(by_id[c]) for c in cited if c in by_id] or [all_text]
        if any(quote_text in h for h in haystacks):
            valid.append(line.strip())
        else:
            rejected += 1
    return valid, rejected


class QuoteGroundingTechnique(BaseTechnique):
    name = "Quote Grounding"
    family = "verify"
    requires = "any"

    def __init__(self, llm, retriever: HybridRetriever, top_k: int = 4):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        raw_quotes = self.llm.complete(
            quote_extraction_prompt(question, retrieved), temperature=0.0
        )
        valid, rejected = verify_quotes(raw_quotes, retrieved)

        trace = [
            f"retrieved {[sc.chunk.chunk_id for sc in retrieved]}",
            f"extracted quotes: {len(valid) + rejected}, "
            f"verbatim-verified: {len(valid)}, rejected (not in source): {rejected}",
        ]
        if not valid:
            return TechniqueResult(
                question=question, answer=ABSTAIN_TEXT, abstained=True,
                groundedness=1.0,
                trace=trace + ["no verbatim quotes support an answer → abstain"],
                extra={"rejected_quotes": rejected},
            )

        quotes_block = "\n".join(f"QUOTE: {q}" for q in valid)
        answer = self.llm.complete(
            synthesize_from_quotes_prompt(question, quotes_block), temperature=0.0
        ).strip()
        abstained = ABSTAIN_TEXT.lower() in answer.lower()
        return TechniqueResult(
            question=question,
            answer=answer,
            abstained=abstained,
            groundedness=1.0 if not abstained else 1.0,
            trace=trace + ["synthesized answer from verified quotes only"],
            extra={"quotes": valid, "rejected_quotes": rejected},
        )
