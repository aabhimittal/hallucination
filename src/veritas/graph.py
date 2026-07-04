"""Graph-RAG: a lightweight knowledge graph over the corpus for multi-hop
retrieval.

Standard vector/lexical retrieval returns flat snippets and struggles with
questions that require *linking* two facts across documents ("which country
borders the country where X is?"). Graph-RAG maps the corpus into entities
(nodes) and the sentences that connect them (edges), then answers by walking
the graph from the question's entities outward, collecting the sentences on the
path as evidence. This keeps distinct topics from being conflated and surfaces
bridging facts a single snippet would miss.

Pure Python (dict-based adjacency) — no graph-DB or embedding dependency, so it
runs offline. Entity extraction is rule-based (capitalized spans + salient
numbers); good enough for the bundled corpus and fully deterministic. A real
deployment would swap in LLM/NER extraction, but the traversal and multi-hop
retrieval below are the reusable part.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from .chunking import Chunk, split_sentences
from .retrieval import STOPWORDS, ScoredChunk, content_tokens

# A candidate entity: a capitalized multi-word span, or a standalone number.
_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+(?:of|the|and)?\s*[A-Z][a-zA-Z]+)*)\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_LEADING_STOP = re.compile(r"^(The|A|An|Its|It|In|On|Of|By)\s+", re.IGNORECASE)


def extract_entities(sentence: str) -> Set[str]:
    ents: Set[str] = set()
    for match in _ENTITY_RE.findall(sentence):
        cleaned = _LEADING_STOP.sub("", match).strip()
        if len(cleaned) > 2 and cleaned.lower() not in STOPWORDS:
            ents.add(cleaned)
    for num in _NUMBER_RE.findall(sentence):
        ents.add(num.replace(",", ""))
    return ents


@dataclass
class SentenceNode:
    chunk_id: str
    text: str
    entities: Set[str] = field(default_factory=set)


@dataclass
class KnowledgeGraph:
    # entity -> sentence indices that mention it
    entity_sentences: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    # entity -> entities co-occurring in some sentence (the graph edges)
    adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    sentences: List[SentenceNode] = field(default_factory=list)

    def entities(self) -> Set[str]:
        return set(self.entity_sentences)

    def match_entities(self, query: str) -> Set[str]:
        """Entities from the graph that appear in the query (case-insensitive)."""
        low = query.lower()
        found = {e for e in self.entity_sentences if e.lower() in low}
        # also match salient content tokens against single-word entities
        q_tokens = set(content_tokens(query))
        for e in self.entity_sentences:
            if " " not in e and e.lower() in q_tokens:
                found.add(e)
        return found

    def neighbors(self, entity: str) -> Set[str]:
        return self.adjacency.get(entity, set())

    def sentences_for(self, entities: Sequence[str]) -> List[int]:
        idxs: Set[int] = set()
        for e in entities:
            idxs |= self.entity_sentences.get(e, set())
        return sorted(idxs)


def build_graph(chunks: Sequence[Chunk]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for chunk in chunks:
        sentences = split_sentences(chunk.text)
        per_sentence = [extract_entities(s) for s in sentences]
        # lightweight coreference: a sentence with a pronoun subject ("Its
        # summit...") inherits the chunk's entities so it stays reachable from
        # the chunk's topic. Edges, below, still come only from real
        # sentence-level co-occurrence to keep the relation graph precise.
        chunk_entities: Set[str] = set().union(*per_sentence) if per_sentence else set()
        for sent, ents in zip(sentences, per_sentence):
            idx = len(graph.sentences)
            reach_ents = ents | chunk_entities
            graph.sentences.append(SentenceNode(chunk.chunk_id, sent, reach_ents))
            for e in reach_ents:
                graph.entity_sentences[e].add(idx)
            # relation edges: only entities that literally co-occur in the
            # same sentence
            ent_list = list(ents)
            for i in range(len(ent_list)):
                for j in range(i + 1, len(ent_list)):
                    graph.adjacency[ent_list[i]].add(ent_list[j])
                    graph.adjacency[ent_list[j]].add(ent_list[i])
    return graph


class GraphRetriever:
    """Multi-hop retriever with the same ``retrieve``/``confidence`` surface as
    :class:`veritas.retrieval.HybridRetriever`, so it drops into any technique.
    """

    def __init__(self, chunks: Sequence[Chunk], hops: int = 2):
        self.chunks = list(chunks)
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.graph = build_graph(chunks)
        self.hops = hops

    def _expand(self, seeds: Set[str]) -> Set[str]:
        """Breadth-first entity expansion up to ``self.hops`` hops."""
        frontier = set(seeds)
        visited = set(seeds)
        for _ in range(self.hops):
            nxt: Set[str] = set()
            for e in frontier:
                nxt |= self.graph.neighbors(e)
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return visited

    def retrieve(self, query: str, k: int = 4) -> List[ScoredChunk]:
        seeds = self.graph.match_entities(query)
        reachable = self._expand(seeds) if seeds else set()
        sent_idxs = self.graph.sentences_for(reachable)

        q_tokens = set(content_tokens(query))
        # score sentences by query-term overlap + a bonus for touching a seed
        scored: List[Tuple[float, int]] = []
        for idx in sent_idxs:
            node = self.graph.sentences[idx]
            overlap = len(q_tokens & set(content_tokens(node.text)))
            seed_bonus = 1.0 if (node.entities & seeds) else 0.0
            hop_terms = len(node.entities & reachable)
            # direct query-term overlap dominates; the hop bonus only breaks
            # ties so multi-hop expansion never floods out the best snippet
            score = 2.0 * overlap + 1.5 * seed_bonus + 0.2 * hop_terms
            scored.append((score, idx))
        scored.sort(reverse=True)

        # aggregate the best sentences back up to their chunks, preserving order
        results: List[ScoredChunk] = []
        seen_chunks: Set[str] = set()
        max_score = scored[0][0] if scored else 1.0
        for score, idx in scored:
            node = self.graph.sentences[idx]
            if node.chunk_id in seen_chunks:
                continue
            seen_chunks.add(node.chunk_id)
            chunk = self._chunk_by_id[node.chunk_id]
            norm = score / max_score if max_score else 0.0
            results.append(
                ScoredChunk(chunk=chunk, score=norm, bm25=score, cosine=norm,
                            coverage=norm)
            )
            if len(results) >= k:
                break
        return results

    def confidence(self, query: str, retrieved: Sequence[ScoredChunk]) -> float:
        """Graph-grounded confidence: did we find seed entities and cover the
        query's content terms with the reachable sentences?"""
        seeds = self.graph.match_entities(query)
        if not seeds or not retrieved:
            return 0.0
        q_content = content_tokens(query)
        if not q_content:
            return 0.0
        evidence_terms: Set[str] = set()
        for sc in retrieved:
            evidence_terms |= set(content_tokens(sc.chunk.text))
        coverage = sum(1 for t in q_content if t in evidence_terms) / len(q_content)
        seed_ratio = min(1.0, len(seeds) / max(1, len(set(q_content)) * 0.4))
        return 0.6 * coverage + 0.4 * seed_ratio
