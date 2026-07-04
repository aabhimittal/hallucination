"""Hallucination-reduction techniques behind one comparable interface.

``build_registry`` returns the set of techniques that run on any LLM (offline
included). White-box decoding techniques (DoLa) live in ``decoding`` and are
added separately because they require a local Hugging Face model.
"""

from __future__ import annotations

from typing import Dict

from .base import BaseTechnique, Technique, TechniqueResult
from .calibration import CalibrationTechnique
from .guardrails import GuardrailsTechnique
from .multi_agent import MultiAgentTechnique
from .nli import HFNLIScorer, LexicalNLI, NLIScorer
from .quote_grounding import QuoteGroundingTechnique
from .semantic_entropy import SemanticEntropyTechnique
from .wrappers import BaselineTechnique, VeritasTechnique

__all__ = [
    "BaseTechnique",
    "BaselineTechnique",
    "CalibrationTechnique",
    "GuardrailsTechnique",
    "HFNLIScorer",
    "LexicalNLI",
    "MultiAgentTechnique",
    "NLIScorer",
    "QuoteGroundingTechnique",
    "SemanticEntropyTechnique",
    "Technique",
    "TechniqueResult",
    "VeritasTechnique",
    "build_registry",
]


def build_registry(llm, retriever, scorer: NLIScorer = None) -> Dict[str, Technique]:
    """All black-box techniques (run on any LLM, including the offline mock).

    Graph-RAG is added by the caller when a graph-backed retriever is built
    (see ``veritas.graph``), and DoLa when a local HF model is available.
    """
    return {
        t.name: t
        for t in [
            BaselineTechnique(llm, retriever),
            VeritasTechnique(llm, retriever),
            SemanticEntropyTechnique(llm, retriever, scorer=scorer),
            QuoteGroundingTechnique(llm, retriever),
            MultiAgentTechnique(llm, retriever, scorer=scorer),
            GuardrailsTechnique(llm, retriever),
            CalibrationTechnique(llm, retriever),
        ]
    }
