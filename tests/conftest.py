import pytest

from veritas import Document, HybridRetriever, chunk_corpus


@pytest.fixture
def corpus_docs():
    return [
        Document(
            "d1",
            "Mount Everest is the highest mountain on Earth. Its summit stands at "
            "8849 meters above sea level. It is located in the Himalayas on the "
            "border of Nepal and China.",
            title="Mount Everest",
        ),
        Document(
            "d2",
            "The Pacific Ocean is the largest ocean on Earth. It covers about 165 "
            "million square kilometers. The Mariana Trench in the Pacific Ocean is "
            "the deepest known point of the world's oceans.",
            title="Pacific Ocean",
        ),
        Document(
            "d3",
            "Photosynthesis is the process by which green plants convert sunlight "
            "into chemical energy. Chlorophyll absorbs light mostly in the blue and "
            "red parts of the spectrum. Oxygen is released as a byproduct of "
            "photosynthesis.",
            title="Photosynthesis",
        ),
    ]


@pytest.fixture
def chunks(corpus_docs):
    return chunk_corpus(corpus_docs)


@pytest.fixture
def retriever(chunks):
    return HybridRetriever(chunks)
