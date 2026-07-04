"""Sentence-aware document chunking with overlap.

Chunks are the retrieval unit for the whole pipeline. Each chunk carries a
stable id (``c1``, ``c2``, ...) that the LLM uses in citations, so chunk ids
must be deterministic for a given corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "Dr.", "Mr.", "Mrs.", "Ms.", "vs.", "St.", "No.")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences with a lightweight regex splitter."""
    text = " ".join(text.split())
    if not text:
        return []
    # Protect common abbreviations from being treated as sentence ends.
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "․"))
    parts = _SENTENCE_RE.split(protected)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


_CITED_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z0-9\"'(])")


def split_cited_sentences(text: str) -> List[str]:
    """Sentence split that keeps trailing ``[cN]`` citation markers attached.

    ``"A boils at 100 C. [c2] B is red. [c3]"`` →
    ``["A boils at 100 C. [c2]", "B is red. [c3]"]``
    """
    text = " ".join(text.split())
    if not text:
        return []
    parts = _CITED_SPLIT_RE.split(text)
    # merge fragments that are pure citation markers into the previous part
    merged: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and re.fullmatch(r"(?:\[c\d+\]\s*)+", part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    sentences: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.chunk_id}] {self.text}"


def chunk_document(
    doc: Document,
    max_sentences: int = 3,
    overlap: int = 1,
    start_index: int = 1,
) -> List[Chunk]:
    """Chunk one document into windows of ``max_sentences`` with ``overlap``.

    ``start_index`` sets the numeric suffix of the first chunk id so that ids
    stay unique across a corpus.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be >= 1")
    if not 0 <= overlap < max_sentences:
        raise ValueError("overlap must satisfy 0 <= overlap < max_sentences")

    sentences = split_sentences(doc.text)
    chunks: List[Chunk] = []
    step = max_sentences - overlap
    i = 0
    n = start_index
    while i < len(sentences):
        window = sentences[i : i + max_sentences]
        chunks.append(
            Chunk(
                chunk_id=f"c{n}",
                doc_id=doc.doc_id,
                text=" ".join(window),
                sentences=list(window),
            )
        )
        n += 1
        if i + max_sentences >= len(sentences):
            break
        i += step
    return chunks


def chunk_corpus(
    docs: Iterable[Document], max_sentences: int = 3, overlap: int = 1
) -> List[Chunk]:
    """Chunk a whole corpus with globally unique, deterministic chunk ids."""
    chunks: List[Chunk] = []
    next_index = 1
    for doc in docs:
        doc_chunks = chunk_document(
            doc, max_sentences=max_sentences, overlap=overlap, start_index=next_index
        )
        chunks.extend(doc_chunks)
        next_index += len(doc_chunks)
    return chunks


def documents_from_texts(texts: Sequence[str], prefix: str = "doc") -> List[Document]:
    """Wrap raw strings (e.g. user-pasted text in the demo) as Documents."""
    return [
        Document(doc_id=f"{prefix}{i + 1}", text=t) for i, t in enumerate(texts) if t.strip()
    ]


def load_documents_from_dir(path) -> List[Document]:
    """Load every ``*.txt`` file in ``path`` as a Document (id = file stem)."""
    from pathlib import Path

    directory = Path(path)
    docs = []
    for file in sorted(directory.glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(doc_id=file.stem, text=text, title=file.stem))
    return docs
