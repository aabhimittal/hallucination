"""Common interface so every hallucination-reduction technique is comparable.

Each technique wraps some combination of an LLM + retriever and exposes a
single ``answer(question) -> TechniqueResult``. The unified benchmark
(``benchmarks/run_comparison.py``) runs every registered technique over the
same dataset with the same independent judge, so the numbers are apples-to-
apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class TechniqueResult:
    question: str
    answer: str
    abstained: bool
    # self-reported / derived confidence in [0,1] (None if the technique
    # doesn't produce one) — consumed by the calibration evaluation
    confidence: Optional[float] = None
    # fraction of the answer supported by evidence (None if not computed)
    groundedness: Optional[float] = None
    # short human-readable stage log for the demo's trace view
    trace: List[str] = field(default_factory=list)
    # technique-specific extras (entropy value, cluster count, quotes, ...)
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Technique(Protocol):
    name: str
    family: str          # "baseline" | "verify" | "uncertainty" | "graph" | "decoding" | "guardrail"
    requires: str        # "any" (works on any LLM incl. offline) | "local_hf"

    def answer(self, question: str) -> TechniqueResult:
        ...


class BaseTechnique:
    """Convenience base with sensible class attributes."""

    name: str = "base"
    family: str = "baseline"
    requires: str = "any"

    def answer(self, question: str) -> TechniqueResult:  # pragma: no cover
        raise NotImplementedError
