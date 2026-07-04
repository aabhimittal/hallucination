import pytest

from veritas.chunking import (
    Document,
    chunk_corpus,
    chunk_document,
    documents_from_texts,
    split_cited_sentences,
    split_sentences,
)


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one? Fourth."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
        "Fourth.",
    ]


def test_split_sentences_protects_abbreviations():
    text = "Dr. Smith studied e.g. rocks. He wrote a paper."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "Dr. Smith studied e.g. rocks."


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_split_cited_sentences_keeps_citations_attached():
    text = "Water boils at 100 C. [c2] The sky is blue. [c3] [c4]"
    parts = split_cited_sentences(text)
    assert parts == ["Water boils at 100 C. [c2]", "The sky is blue. [c3] [c4]"]


def test_split_cited_sentences_decimal_numbers_not_split():
    text = "It covers 165.5 million km. [c1] It is deep. [c2]"
    parts = split_cited_sentences(text)
    assert parts[0] == "It covers 165.5 million km. [c1]"
    assert len(parts) == 2


def test_chunk_document_windows_and_overlap():
    doc = Document("d1", "One a. Two b. Three c. Four d. Five e.")
    chunks = chunk_document(doc, max_sentences=3, overlap=1)
    assert [c.chunk_id for c in chunks] == ["c1", "c2"]
    assert chunks[0].sentences == ["One a.", "Two b.", "Three c."]
    # overlap of 1: second window starts at the last sentence of the first
    assert chunks[1].sentences[0] == "Three c."
    # all sentences are covered
    covered = set(s for c in chunks for s in c.sentences)
    assert covered == {"One a.", "Two b.", "Three c.", "Four d.", "Five e."}


def test_chunk_document_invalid_params():
    doc = Document("d1", "A b. C d.")
    with pytest.raises(ValueError):
        chunk_document(doc, max_sentences=0)
    with pytest.raises(ValueError):
        chunk_document(doc, max_sentences=2, overlap=2)


def test_chunk_corpus_unique_global_ids(corpus_docs):
    chunks = chunk_corpus(corpus_docs)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "c1"
    # ids are sequential
    assert ids == [f"c{i + 1}" for i in range(len(ids))]
    # every doc contributed at least one chunk
    assert {c.doc_id for c in chunks} == {"d1", "d2", "d3"}


def test_documents_from_texts_skips_blank():
    docs = documents_from_texts(["hello world.", "  ", "second doc."])
    assert [d.doc_id for d in docs] == ["doc1", "doc3"]
