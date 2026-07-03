import pytest

from veritas.retrieval import HybridRetriever, content_tokens, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Mount Everest, 8849m!") == ["mount", "everest", "8849m"]


def test_content_tokens_removes_stopwords():
    toks = content_tokens("What is the height of Mount Everest?")
    assert "the" not in toks
    assert "what" not in toks
    assert "everest" in toks


def test_retrieve_ranks_relevant_chunk_first(retriever):
    results = retriever.retrieve("How tall is Mount Everest?", k=3)
    assert results[0].chunk.doc_id == "d1"
    assert results[0].score >= results[-1].score


def test_retrieve_respects_k(retriever):
    assert len(retriever.retrieve("ocean", k=2)) == 2


def test_scores_are_bounded(retriever):
    for sc in retriever.retrieve("photosynthesis chlorophyll light", k=5):
        assert 0.0 <= sc.score <= 1.0 + 1e-9
        assert 0.0 <= sc.coverage <= 1.0


def test_confidence_high_for_answerable_question(retriever):
    query = "What is the height of Mount Everest?"
    retrieved = retriever.retrieve(query, k=3)
    assert retriever.confidence(query, retrieved) > 0.55


def test_confidence_low_for_unanswerable_question(retriever):
    query = "Who won the 1994 FIFA World Cup final in Pasadena?"
    retrieved = retriever.retrieve(query, k=3)
    assert retriever.confidence(query, retrieved) < 0.45


def test_confidence_empty_inputs(retriever):
    assert retriever.confidence("anything", []) == 0.0
    retrieved = retriever.retrieve("of the and", k=2)
    # a stopword-only query carries no content signal
    assert retriever.confidence("of the and", retrieved) == 0.0


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([])
