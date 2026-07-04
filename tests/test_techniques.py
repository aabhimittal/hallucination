import pytest

from veritas import HybridRetriever, MockLLM
from veritas.techniques import build_registry
from veritas.techniques.calibration import CalibrationTechnique, parse_confidence
from veritas.techniques.guardrails import GuardrailsTechnique, check_output_rails
from veritas.techniques.multi_agent import MultiAgentTechnique
from veritas.techniques.nli import LexicalNLI, bidirectional_equivalent
from veritas.techniques.quote_grounding import QuoteGroundingTechnique, verify_quotes
from veritas.techniques.semantic_entropy import (
    SemanticEntropyTechnique,
    cluster_by_meaning,
    semantic_entropy,
)

ANSWERABLE = "What is the height of the summit of Mount Everest in meters?"
UNANSWERABLE = "Who won the 1994 FIFA World Cup?"


def test_registry_has_all_black_box_techniques(retriever):
    reg = build_registry(MockLLM(), retriever)
    assert set(reg) == {
        "Baseline RAG", "VERITAS", "Semantic Entropy", "Quote Grounding",
        "Multi-Agent Consensus", "Neurosymbolic Guardrails",
        "Calibrated Selective Prediction",
    }
    for tech in reg.values():
        assert tech.requires == "any"


# ------------------------------------------------------------ semantic entropy


def test_semantic_clustering_and_entropy():
    scorer = LexicalNLI()
    answers = [
        "Everest is 8849 meters tall.",
        "The summit of Everest stands at 8849 meters.",  # same meaning
        "The Nile is the longest river in Africa.",       # different
    ]
    clusters = cluster_by_meaning(answers, scorer, threshold=0.5)
    assert len(clusters) == 2
    ent = semantic_entropy(clusters)
    assert ent > 0
    # a single cluster has zero entropy
    assert semantic_entropy([answers]) == 0.0


def test_bidirectional_equivalence():
    scorer = LexicalNLI()
    assert bidirectional_equivalent(
        "Everest is the highest mountain.",
        "Everest is the highest mountain.",
        scorer,
    )
    assert not bidirectional_equivalent(
        "Everest is the highest mountain.",
        "Bananas are yellow fruit.",
        scorer,
    )


def test_semantic_entropy_answers_consistent_question(retriever):
    # clean model → all samples agree → low entropy → answered
    tech = SemanticEntropyTechnique(MockLLM(hallucination_rate=0.0), retriever)
    result = tech.answer(ANSWERABLE)
    assert not result.abstained
    assert result.extra["entropy"] == pytest.approx(0.0)
    assert "8849" in result.answer


def test_semantic_entropy_abstains_on_guessing(retriever):
    # unanswerable → each sample fabricates differently → high entropy → abstain
    tech = SemanticEntropyTechnique(MockLLM(hallucination_rate=0.0), retriever)
    result = tech.answer(UNANSWERABLE)
    assert result.abstained
    assert result.confidence is not None and result.confidence < 0.5


# -------------------------------------------------------------- quote grounding


def test_verify_quotes_rejects_non_substring(retriever):
    retrieved = retriever.retrieve("Mount Everest", k=2)
    raw = (
        "QUOTE: Mount Everest is the highest mountain on Earth. [c1]\n"
        "QUOTE: Everest was first climbed by aliens in 1823. [c1]"
    )
    valid, rejected = verify_quotes(raw, retrieved)
    assert rejected == 1
    assert len(valid) == 1
    assert "highest mountain" in valid[0]


def test_quote_grounding_answers_and_abstains(retriever):
    tech = QuoteGroundingTechnique(MockLLM(), retriever)
    assert not tech.answer(ANSWERABLE).abstained
    assert tech.answer(UNANSWERABLE).abstained


# --------------------------------------------------------------- guardrails


def test_output_rails_flag_missing_citation_and_speculation():
    ok, violations = check_output_rails("Everest is 8849 m tall. [c1]")
    assert ok and not violations

    bad_cite, v1 = check_output_rails("Everest is 8849 m tall.")
    assert not bad_cite and any("citation" in v for v in v1)

    speculative, v2 = check_output_rails("I think Everest is probably tall. [c1]")
    assert not speculative and any("speculation" in v for v in v2)


def test_guardrails_input_rail_refuses_out_of_scope(retriever):
    tech = GuardrailsTechnique(MockLLM(), retriever)
    result = tech.answer(UNANSWERABLE)
    assert result.abstained
    assert any("input rail" in line for line in result.trace)


def test_guardrails_answers_in_scope(retriever):
    tech = GuardrailsTechnique(MockLLM(), retriever)
    result = tech.answer(ANSWERABLE)
    assert not result.abstained


# ------------------------------------------------------------- multi-agent


def test_multi_agent_answers_and_abstains(retriever):
    tech = MultiAgentTechnique(MockLLM(), retriever)
    ans = tech.answer(ANSWERABLE)
    assert not ans.abstained
    assert any("researcher" in line for line in ans.trace)
    assert any("judge" in line for line in ans.trace)
    assert tech.answer(UNANSWERABLE).abstained


def test_multi_agent_drops_unsupported_claim(retriever):
    class OneBadClaimLLM(MockLLM):
        def _answer(self, prompt, grounded, nonce=0):
            return (
                "Mount Everest is the highest mountain on Earth. [c1] "
                "Everest is made entirely of chocolate. [c1]"
            )

    tech = MultiAgentTechnique(OneBadClaimLLM(), retriever)
    result = tech.answer(ANSWERABLE)
    assert not result.abstained
    assert "chocolate" not in result.answer
    assert "highest mountain" in result.answer


# ------------------------------------------------------------- calibration


def test_parse_confidence():
    assert parse_confidence("answer\nCONFIDENCE: 0.9") == 0.9
    assert parse_confidence("CONFIDENCE: 1.0") == 1.0
    assert parse_confidence("no confidence here") is None


def test_calibration_technique_reports_confidence(retriever):
    tech = CalibrationTechnique(MockLLM(), retriever)
    result = tech.answer(ANSWERABLE)
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert "CONFIDENCE" not in result.answer  # stripped from the visible answer


def test_calibration_selective_prediction_withholds_low_confidence(retriever):
    # unanswerable → low stated confidence → withheld
    tech = CalibrationTechnique(MockLLM(), retriever, confidence_threshold=0.5)
    result = tech.answer(UNANSWERABLE)
    assert result.abstained
