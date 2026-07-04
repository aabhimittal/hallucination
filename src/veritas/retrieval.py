"""Hybrid lexical retrieval (BM25 + TF-IDF cosine) with a confidence gate.

Pure Python by design: no numpy/faiss/embedding models, so the test suite and
the Hugging Face Space run instantly with zero downloads. The retriever also
produces a calibrated *confidence* signal used by the pipeline's abstention
gate — the first line of defense against hallucination is refusing to answer
questions the corpus cannot support.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with does did do can could
    should would about into over under between many much more most""".split()
)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float          # fused hybrid score in [0, 1]
    bm25: float
    cosine: float
    coverage: float       # fraction of query content-terms present in the chunk


class HybridRetriever:
    """BM25 + TF-IDF-cosine retriever over a fixed chunk list."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = [tokenize(c.text) for c in self.chunks]
        self._doc_counters: List[Counter] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avg_len = sum(self._doc_lens) / len(self._doc_lens)
        self._n = len(self.chunks)
        # document frequency per term
        self._df: Dict[str, int] = Counter()
        for counter in self._doc_counters:
            for term in counter:
                self._df[term] += 1

    # ---------------------------------------------------------------- BM25
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def _bm25(self, query_tokens: List[str], idx: int) -> float:
        score = 0.0
        counter = self._doc_counters[idx]
        dl = self._doc_lens[idx] or 1
        for term in query_tokens:
            tf = counter.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
            score += idf * tf * (self.k1 + 1) / denom
        return score

    # -------------------------------------------------------------- cosine
    def _tfidf_vec(self, counter: Counter) -> Dict[str, float]:
        vec = {}
        for term, tf in counter.items():
            df = self._df.get(term, 0)
            idf = math.log((self._n + 1) / (df + 1)) + 1
            vec[term] = tf * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ------------------------------------------------------------ retrieve
    def retrieve(self, query: str, k: int = 4) -> List[ScoredChunk]:
        q_tokens = tokenize(query)
        q_content = content_tokens(query)
        q_vec = self._tfidf_vec(Counter(q_tokens))

        bm25_scores = [self._bm25(q_tokens, i) for i in range(self._n)]
        max_bm25 = max(bm25_scores) or 1.0
        results: List[ScoredChunk] = []
        for i, chunk in enumerate(self.chunks):
            cos = self._cosine(q_vec, self._tfidf_vec(self._doc_counters[i]))
            chunk_terms = set(self._doc_counters[i])
            coverage = (
                sum(1 for t in q_content if t in chunk_terms) / len(q_content)
                if q_content
                else 0.0
            )
            bm25_norm = bm25_scores[i] / max_bm25
            fused = 0.5 * bm25_norm + 0.3 * cos + 0.2 * coverage
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=fused,
                    bm25=bm25_scores[i],
                    cosine=cos,
                    coverage=coverage,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def confidence(self, query: str, retrieved: Sequence[ScoredChunk]) -> float:
        """Evidence confidence in [0, 1] for the abstention gate.

        Combines how much of the query's content vocabulary the evidence set
        actually covers (the dominant signal for unanswerable questions) with
        the strength of the best cosine match. BM25 is deliberately excluded
        here because its per-query normalization makes even weak matches look
        strong.
        """
        if not retrieved:
            return 0.0
        q_content = content_tokens(query)
        if not q_content:
            return 0.0
        evidence_terms = set()
        for sc in retrieved:
            evidence_terms.update(tokenize(sc.chunk.text))
        joint_coverage = sum(1 for t in q_content if t in evidence_terms) / len(q_content)
        best_cosine = max(sc.cosine for sc in retrieved)
        return 0.65 * joint_coverage + 0.35 * min(1.0, best_cosine * 2.0)
