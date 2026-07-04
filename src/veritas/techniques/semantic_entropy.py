"""Semantic Entropy (Kuhn/Farquhar/Gal — Nature 2024) as an abstention gate.

Sample the same question several times at non-zero temperature, cluster the
answers by *meaning* (bidirectional entailment), and measure the entropy of the
meaning distribution. High semantic entropy = the model is guessing (its
answers disagree about what's true) → abstain. Unlike naive token entropy this
ignores paraphrase: "8849 m" and "8,849 metres" fall in one cluster.

Black-box: needs only sampled generations, so it works on any LLM.
"""

from __future__ import annotations

import math
from typing import List

from ..prompts import ABSTAIN_TEXT, grounded_answer_prompt
from ..retrieval import HybridRetriever
from .base import BaseTechnique, TechniqueResult
from .nli import LexicalNLI, NLIScorer, bidirectional_equivalent


def cluster_by_meaning(answers: List[str], scorer: NLIScorer, threshold: float = 0.6):
    """Greedy semantic clustering: an answer joins the first cluster whose
    representative it is mutually-entailing with, else it seeds a new cluster."""
    clusters: List[List[str]] = []
    for ans in answers:
        placed = False
        for cluster in clusters:
            if bidirectional_equivalent(ans, cluster[0], scorer, threshold):
                cluster.append(ans)
                placed = True
                break
        if not placed:
            clusters.append([ans])
    return clusters


def semantic_entropy(clusters: List[List[str]]) -> float:
    """Shannon entropy (nats) over the cluster-size distribution."""
    n = sum(len(c) for c in clusters)
    if n == 0:
        return 0.0
    ent = 0.0
    for cluster in clusters:
        p = len(cluster) / n
        ent -= p * math.log(p)
    return ent


class SemanticEntropyTechnique(BaseTechnique):
    name = "Semantic Entropy"
    family = "uncertainty"
    requires = "any"

    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        n_samples: int = 6,
        temperature: float = 0.7,
        entropy_threshold: float = 0.9,
        scorer: NLIScorer = None,
        top_k: int = 4,
    ):
        self.llm = llm
        self.retriever = retriever
        self.n_samples = n_samples
        self.temperature = temperature
        self.entropy_threshold = entropy_threshold
        self.scorer = scorer or LexicalNLI()
        self.top_k = top_k

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        prompt = grounded_answer_prompt(question, retrieved)
        samples = [
            self.llm.complete(prompt, temperature=self.temperature).strip()
            for _ in range(self.n_samples)
        ]
        # answers where the model itself abstained don't vote for a meaning
        contentful = [s for s in samples if ABSTAIN_TEXT.lower() not in s.lower()]
        abstain_votes = len(samples) - len(contentful)

        if not contentful:
            return self._abstain(question, samples, 0.0, [])

        clusters = cluster_by_meaning(contentful, self.scorer)
        entropy = semantic_entropy(clusters)
        # confidence falls as entropy rises; normalized by log(n_samples)
        norm = math.log(self.n_samples) or 1.0
        confidence = max(0.0, 1.0 - entropy / norm)

        if entropy > self.entropy_threshold or abstain_votes > len(samples) / 2:
            return self._abstain(question, samples, entropy, clusters, confidence)

        # answer = the majority meaning cluster's first sample
        best = max(clusters, key=len)
        return TechniqueResult(
            question=question,
            answer=best[0],
            abstained=False,
            confidence=confidence,
            trace=[
                f"sampled {self.n_samples} answers at T={self.temperature}",
                f"{len(clusters)} meaning cluster(s), semantic entropy {entropy:.2f} "
                f"(threshold {self.entropy_threshold})",
                f"majority cluster size {len(best)}",
            ],
            extra={"entropy": entropy, "clusters": len(clusters), "samples": samples},
        )

    def _abstain(self, question, samples, entropy, clusters, confidence=0.0):
        return TechniqueResult(
            question=question,
            answer=ABSTAIN_TEXT,
            abstained=True,
            confidence=confidence,
            trace=[
                f"sampled {len(samples)} answers",
                f"semantic entropy {entropy:.2f} above threshold "
                f"{self.entropy_threshold} → model is guessing → abstain",
            ],
            extra={"entropy": entropy, "clusters": len(clusters), "samples": samples},
        )
