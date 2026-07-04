from veritas import (
    ABSTAIN_TEXT,
    BaselineRAG,
    MockLLM,
    PipelineConfig,
    VeritasPipeline,
)
from veritas.metrics import is_abstention
from veritas.verification import Verdict


ANSWERABLE = "What is the height of the summit of Mount Everest in meters?"
UNANSWERABLE = "Who won the 1994 FIFA World Cup final in Pasadena?"


def test_answerable_question_grounded_answer(retriever):
    pipeline = VeritasPipeline(MockLLM(), retriever)
    result = pipeline.answer(ANSWERABLE)
    assert not result.abstained
    assert "8849" in result.answer
    assert "[c" in result.answer  # citations present
    assert result.groundedness == 1.0
    assert result.confidence >= 0.45
    assert [t.stage for t in result.trace][0] == "retrieve"
    assert result.citations  # at least one distinct citation id


def test_unanswerable_question_abstains_at_gate(retriever):
    pipeline = VeritasPipeline(MockLLM(), retriever)
    result = pipeline.answer(UNANSWERABLE)
    assert result.abstained
    assert result.answer == ABSTAIN_TEXT
    assert result.abstain_reason == "low retrieval confidence"
    # the gate fires before any LLM call
    assert result.draft == ""


def test_model_level_abstention_when_gate_disabled(retriever):
    config = PipelineConfig(confidence_threshold=0.0)
    llm = MockLLM()
    pipeline = VeritasPipeline(llm, retriever, config)
    result = pipeline.answer(UNANSWERABLE)
    assert result.abstained
    assert result.abstain_reason == "model abstained"
    assert "generate" in llm.calls


def test_injected_hallucination_is_removed(retriever):
    llm = MockLLM(hallucination_rate=1.0)
    pipeline = VeritasPipeline(llm, retriever)
    result = pipeline.answer(ANSWERABLE)
    assert not result.abstained
    # the draft contains a fabrication, the final answer does not
    assert len(result.draft_verdicts) > len(result.final_verdicts)
    assert result.removed + result.repaired >= 1
    assert all(v.label != Verdict.UNSUPPORTED for v in result.final_verdicts)
    assert result.groundedness == 1.0


def test_baseline_passes_hallucination_through(retriever):
    llm = MockLLM(hallucination_rate=1.0)
    baseline = BaselineRAG(llm, retriever)
    result = baseline.answer(ANSWERABLE)
    # baseline output is the raw draft: fabrication survives
    from veritas.verification import lexical_entailment
    corpus_text = " ".join(c.text for c in retriever.chunks)
    from veritas.claims import split_into_claims_fallback
    scores = [
        lexical_entailment(c.text, corpus_text)
        for c in split_into_claims_fallback(result.answer)
    ]
    assert any(s < 0.4 for s in scores)


def test_baseline_fabricates_on_unanswerable(retriever):
    baseline = BaselineRAG(MockLLM(), retriever)
    result = baseline.answer(UNANSWERABLE)
    assert not is_abstention(result.answer)


def test_mostly_unsupported_answer_downgraded_to_abstention(retriever):
    class FabricatingLLM(MockLLM):
        def _answer(self, prompt, grounded, nonce=0):
            return (
                "The Moon is made of green cheese. [c1] "
                "Napoleon owned a pet dragon named Fluffy. [c1]"
            )

        def _repair(self, prompt):
            return "REMOVE"

    pipeline = VeritasPipeline(FabricatingLLM(), retriever)
    result = pipeline.answer(ANSWERABLE)
    assert result.abstained
    assert result.abstain_reason in ("no verifiable claims", "answer mostly unsupported")


def test_repair_path_replaces_claim(retriever):
    class OneBadClaimLLM(MockLLM):
        def _answer(self, prompt, grounded, nonce=0):
            # one supported claim + one wrong-number claim on the same topic
            return (
                "Mount Everest is the highest mountain on Earth. [c1] "
                "Its summit stands at 9021 meters above sea level. [c1]"
            )

    llm = OneBadClaimLLM()
    pipeline = VeritasPipeline(llm, retriever)
    result = pipeline.answer(ANSWERABLE)
    assert not result.abstained
    assert result.repaired == 1
    assert "8849" in result.answer
    assert "9021" not in result.answer
    assert "repair" in llm.calls


def test_repair_disabled_removes_instead(retriever):
    class OneBadClaimLLM(MockLLM):
        def _answer(self, prompt, grounded, nonce=0):
            return (
                "Mount Everest is the highest mountain on Earth. [c1] "
                "Its summit stands at 9021 meters above sea level. [c1]"
            )

    config = PipelineConfig(repair=False)
    pipeline = VeritasPipeline(OneBadClaimLLM(), retriever, config)
    result = pipeline.answer(ANSWERABLE)
    assert not result.abstained
    assert result.repaired == 0
    assert result.removed == 1
    assert "9021" not in result.answer


def test_lexical_only_verification_mode(retriever):
    config = PipelineConfig(use_llm_verifier=False)
    llm = MockLLM()
    pipeline = VeritasPipeline(llm, retriever, config)
    result = pipeline.answer(ANSWERABLE)
    assert not result.abstained
    assert "verify" not in llm.calls
    assert all(v.llm_verdict is None for v in result.draft_verdicts)


def test_determinism_same_seed_same_output(retriever):
    r1 = VeritasPipeline(MockLLM(hallucination_rate=0.5, seed=7), retriever).answer(ANSWERABLE)
    r2 = VeritasPipeline(MockLLM(hallucination_rate=0.5, seed=7), retriever).answer(ANSWERABLE)
    assert r1.answer == r2.answer
    assert r1.draft == r2.draft
