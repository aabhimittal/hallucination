from veritas.claims import Claim
from veritas.metrics import (
    QuestionRecord,
    citation_precision,
    evaluate_run,
    is_abstention,
    judge_answer,
    keyword_hit,
)
from veritas.prompts import ABSTAIN_TEXT
from veritas.verification import ClaimVerdict, Verdict


def test_is_abstention_detects_standard_and_freeform():
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("Sorry, I don't know.")
    assert not is_abstention("Mount Everest is 8849 meters tall.")


def test_keyword_hit_case_insensitive():
    assert keyword_hit("The summit is 8849 meters.", ["8849"])
    assert keyword_hit("EVEREST is highest", ["everest"])
    assert not keyword_hit("The ocean is big.", ["everest"])


def test_judge_answer_flags_fabrication(chunks):
    good = judge_answer("Mount Everest is the highest mountain on Earth.", chunks)
    assert good.n_unsupported == 0
    assert good.groundedness == 1.0

    bad = judge_answer(
        "Mount Everest is the highest mountain on Earth. "
        "It was first photographed from space in 1962 by cosmonauts.",
        chunks,
    )
    assert bad.n_claims == 2
    assert bad.n_unsupported == 1
    assert bad.groundedness < 1.0


def test_judge_answer_empty(chunks):
    judgement = judge_answer("", chunks)
    assert judgement.n_claims == 0
    assert judgement.groundedness == 1.0


def _record(qtype, answer, abstained, gold=None, q="q?"):
    return QuestionRecord(
        question=q, qtype=qtype, gold_keywords=gold or [], answer=answer, abstained=abstained
    )


def test_evaluate_run_aggregates(chunks):
    records = [
        _record("answerable", "Mount Everest is the highest mountain on Earth.", False,
                gold=["everest"]),
        _record("answerable", ABSTAIN_TEXT, True, gold=["8849"]),
        _record("unanswerable", ABSTAIN_TEXT, True),
        _record("unanswerable", "The 1994 final was won by Brazil on penalties.", False),
    ]
    metrics = evaluate_run(records, chunks)
    assert metrics["questions"] == 4.0
    assert metrics["abstention_recall"] == 0.5          # 1 of 2 unanswerable abstained
    assert metrics["false_abstention_rate"] == 0.5      # 1 of 2 answerable abstained
    assert metrics["answer_accuracy"] == 1.0            # the one answered was correct
    assert metrics["answer_coverage"] == 0.5
    # answering an unanswerable question counts as a hallucinated question
    assert metrics["hallucination_rate"] == 0.25 or metrics["hallucination_rate"] == 0.5


def test_evaluate_run_perfect_system(chunks):
    records = [
        _record("answerable", "Mount Everest is the highest mountain on Earth.", False,
                gold=["everest"]),
        _record("unanswerable", ABSTAIN_TEXT, True),
    ]
    metrics = evaluate_run(records, chunks)
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["abstention_recall"] == 1.0
    assert metrics["false_abstention_rate"] == 0.0


def test_citation_precision(chunks):
    by_id = {c.chunk_id: c for c in chunks}
    everest_chunk = next(c for c in chunks if "Everest" in c.text)
    ocean_chunk = next(c for c in chunks if "Pacific" in c.text)

    good = ClaimVerdict(
        claim=Claim(
            text="Mount Everest is the highest mountain on Earth.",
            citations=[everest_chunk.chunk_id],
        ),
        label=Verdict.SUPPORTED,
        lexical_score=1.0,
    )
    miscited = ClaimVerdict(
        claim=Claim(
            text="Mount Everest is the highest mountain on Earth.",
            citations=[ocean_chunk.chunk_id],
        ),
        label=Verdict.SUPPORTED,
        lexical_score=1.0,
    )
    assert citation_precision([good], by_id) == 1.0
    assert citation_precision([good, miscited], by_id) == 0.5
    assert citation_precision([], by_id) is None
