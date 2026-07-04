from veritas.claims import (
    decompose_answer,
    parse_claim_line,
    split_into_claims_fallback,
)
from veritas.llm import MockLLM


def test_parse_claim_line_numbered_with_citation():
    claim = parse_claim_line("1. Mount Everest is 8849 meters tall. [c2]")
    assert claim.text == "Mount Everest is 8849 meters tall."
    assert claim.citations == ["c2"]
    assert "[c2]" in claim.raw


def test_parse_claim_line_multiple_citations():
    claim = parse_claim_line("2) The ocean is large. [c1] [c3]")
    assert claim.citations == ["c1", "c3"]


def test_parse_claim_line_blank_returns_none():
    assert parse_claim_line("") is None
    assert parse_claim_line("3. [c1]") is None  # citation-only line has no claim


def test_split_into_claims_fallback_sentence_per_claim():
    answer = "Everest is the highest mountain. [c1] Its summit is 8849 meters. [c1]"
    claims = split_into_claims_fallback(answer)
    assert len(claims) == 2
    assert claims[0].citations == ["c1"]
    assert claims[1].text == "Its summit is 8849 meters."


def test_decompose_answer_uses_llm():
    llm = MockLLM()
    answer = "Everest is the highest mountain. [c1] Its summit is 8849 meters. [c2]"
    claims = decompose_answer(answer, llm=llm)
    assert "decompose" in llm.calls
    assert [c.citations for c in claims] == [["c1"], ["c2"]]


def test_decompose_answer_falls_back_without_llm():
    answer = "One fact. Another fact."
    claims = decompose_answer(answer, llm=None)
    assert [c.text for c in claims] == ["One fact.", "Another fact."]


class _GarbageLLM:
    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        return "\n\n   \n"  # unusable output


def test_decompose_answer_falls_back_on_unusable_llm_output():
    claims = decompose_answer("A clear fact.", llm=_GarbageLLM())
    assert [c.text for c in claims] == ["A clear fact."]
