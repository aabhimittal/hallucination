from veritas.claims import Claim
from veritas.llm import MockLLM
from veritas.verification import (
    Verdict,
    _fuse,
    lexical_entailment,
    parse_llm_verdict,
    verify_answer,
    verify_claim,
)


EVIDENCE = (
    "Mount Everest is the highest mountain on Earth. Its summit stands at "
    "8849 meters above sea level."
)


def test_lexical_entailment_supported():
    assert lexical_entailment("Mount Everest is the highest mountain.", EVIDENCE) >= 0.9


def test_lexical_entailment_unsupported():
    score = lexical_entailment("The Nile is the longest river in Africa.", EVIDENCE)
    assert score < 0.4


def test_lexical_entailment_number_mismatch_penalized():
    right = lexical_entailment("The summit stands at 8849 meters.", EVIDENCE)
    wrong = lexical_entailment("The summit stands at 9021 meters.", EVIDENCE)
    assert right >= 0.9
    assert wrong <= 0.3


def test_lexical_entailment_empty_claim():
    assert lexical_entailment("", EVIDENCE) == 0.0


def test_parse_llm_verdict():
    assert parse_llm_verdict("analysis...\nVERDICT: SUPPORTED") == Verdict.SUPPORTED
    assert parse_llm_verdict("VERDICT: PARTIAL") == Verdict.PARTIAL
    assert parse_llm_verdict("no verdict here") == Verdict.UNSUPPORTED


def test_fuse_agreement_and_disagreement():
    assert _fuse(Verdict.SUPPORTED, Verdict.SUPPORTED) == Verdict.SUPPORTED
    assert _fuse(Verdict.UNSUPPORTED, Verdict.UNSUPPORTED) == Verdict.UNSUPPORTED
    # disagreement never yields full support
    assert _fuse(Verdict.SUPPORTED, Verdict.PARTIAL) == Verdict.PARTIAL
    assert _fuse(Verdict.SUPPORTED, Verdict.UNSUPPORTED) == Verdict.PARTIAL
    assert _fuse(Verdict.PARTIAL, Verdict.UNSUPPORTED) == Verdict.UNSUPPORTED
    # lexical judge alone when no LLM verdict
    assert _fuse(Verdict.SUPPORTED, None) == Verdict.SUPPORTED


def test_verify_claim_supported(retriever):
    retrieved = retriever.retrieve("Mount Everest height", k=2)
    claim = Claim(text="Mount Everest is the highest mountain on Earth.", citations=[])
    verdict = verify_claim(claim, retrieved, llm=MockLLM())
    assert verdict.label == Verdict.SUPPORTED
    assert verdict.llm_verdict == Verdict.SUPPORTED
    assert verdict.lexical_score >= 0.9


def test_verify_claim_fabricated_number(retriever):
    retrieved = retriever.retrieve("Mount Everest height", k=2)
    claim = Claim(text="Mount Everest stands at 9999 meters above sea level.", citations=[])
    verdict = verify_claim(claim, retrieved, llm=MockLLM())
    assert verdict.label == Verdict.UNSUPPORTED


def test_verify_claim_prefers_cited_chunks(retriever):
    retrieved = retriever.retrieve("Mount Everest height meters", k=4)
    cited_id = retrieved[0].chunk.chunk_id
    claim = Claim(
        text="Mount Everest is the highest mountain on Earth.",
        citations=[cited_id],
    )
    verdict = verify_claim(claim, retrieved, llm=None)
    assert verdict.evidence_ids == [cited_id]


def test_verify_claim_unknown_citation_falls_back_to_all(retriever):
    retrieved = retriever.retrieve("Mount Everest", k=2)
    claim = Claim(text="Mount Everest is the highest mountain.", citations=["c999"])
    verdict = verify_claim(claim, retrieved, llm=None)
    assert len(verdict.evidence_ids) == len(retrieved)


def test_verify_answer_returns_one_verdict_per_claim(retriever):
    retrieved = retriever.retrieve("Mount Everest", k=2)
    claims = [
        Claim(text="Mount Everest is the highest mountain on Earth."),
        Claim(text="Bananas are rich in potassium."),
    ]
    verdicts = verify_answer(claims, retrieved, llm=MockLLM())
    assert len(verdicts) == 2
    assert verdicts[0].label == Verdict.SUPPORTED
    assert verdicts[1].label == Verdict.UNSUPPORTED
