import pytest

from veritas.metrics import (
    QuestionRecord,
    auroc,
    calibration_report,
    expected_calibration_error,
    record_is_correct,
    risk_coverage,
)
from veritas.prompts import ABSTAIN_TEXT


def test_ece_perfect_calibration():
    # confidence exactly matches accuracy in each region → ECE ~ 0
    pairs = [(1.0, True), (1.0, True), (0.0, False), (0.0, False)]
    assert expected_calibration_error(pairs) == pytest.approx(0.0)


def test_ece_overconfident():
    # always 90% confident but only 50% correct → ECE = 0.4
    pairs = [(0.9, True), (0.9, False), (0.9, True), (0.9, False)]
    assert expected_calibration_error(pairs) == pytest.approx(0.4)


def test_ece_empty():
    assert expected_calibration_error([]) is None
    assert expected_calibration_error([(None, True)]) is None


def test_auroc_perfect_and_random():
    # correct answers always more confident than wrong ones → AUROC 1.0
    assert auroc([(0.9, True), (0.8, True), (0.2, False), (0.1, False)]) == 1.0
    # confidence unrelated to correctness → AUROC 0.5 (ties)
    assert auroc([(0.5, True), (0.5, False)]) == 0.5
    # needs both classes
    assert auroc([(0.9, True), (0.8, True)]) is None


def test_risk_coverage_monotone_intuition():
    # most-confident answers are the correct ones → risk stays 0 until the
    # wrong low-confidence answer is included at full coverage
    pairs = [(0.9, True), (0.7, True), (0.2, False)]
    points, aurc = risk_coverage(pairs)
    assert points[0] == (pytest.approx(1 / 3), 0.0)   # top-1: no error
    assert points[-1][0] == pytest.approx(1.0)         # full coverage
    assert points[-1][1] == pytest.approx(1 / 3)       # one of three wrong
    assert 0.0 <= aurc <= 1.0


def test_risk_coverage_empty():
    points, aurc = risk_coverage([])
    assert points == [] and aurc is None


def _rec(qtype, answer, abstained, conf, gold=None):
    return QuestionRecord(
        question="q", qtype=qtype, gold_keywords=gold or [], answer=answer,
        abstained=abstained, confidence=conf,
    )


def test_record_is_correct(chunks):
    # answerable, correct grounded answer with the gold keyword
    good = _rec("answerable", "Mount Everest is the highest mountain on Earth.",
                False, 0.9, gold=["everest"])
    assert record_is_correct(good, chunks)
    # answerable but abstained → wrong
    assert not record_is_correct(
        _rec("answerable", ABSTAIN_TEXT, True, 0.1, gold=["everest"]), chunks
    )
    # unanswerable, abstained → correct
    assert record_is_correct(_rec("unanswerable", ABSTAIN_TEXT, True, 0.1), chunks)
    # unanswerable, answered → wrong
    assert not record_is_correct(
        _rec("unanswerable", "Brazil won in 1994.", False, 0.8), chunks
    )


def test_calibration_report_shapes(chunks):
    records = [
        _rec("answerable", "Mount Everest is the highest mountain on Earth.",
             False, 0.9, gold=["everest"]),
        _rec("unanswerable", ABSTAIN_TEXT, True, 0.1),
        _rec("answerable", "The summit stands at 9999 meters.", False, 0.8,
             gold=["8849"]),  # confident but wrong
    ]
    report = calibration_report(records, chunks)
    assert report["n_with_confidence"] == 3.0
    assert 0.0 <= report["ece"] <= 1.0
    assert report["auroc"] is None or 0.0 <= report["auroc"] <= 1.0
    assert 0.0 <= report["aurc"] <= 1.0


def test_calibration_report_ignores_missing_confidence(chunks):
    records = [_rec("answerable", "x", False, None, gold=["x"])]
    report = calibration_report(records, chunks)
    assert report["n_with_confidence"] == 0.0
    assert report["ece"] is None
