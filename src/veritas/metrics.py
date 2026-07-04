"""Benchmark metrics: quantify hallucination instead of eyeballing it.

The judge here is model-free (lexical entailment against the *entire* corpus,
not just the retrieved evidence) so both systems under test — BaselineRAG and
VeritasPipeline — are graded by the same independent ruler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .chunking import Chunk
from .claims import split_into_claims_fallback
from .prompts import ABSTAIN_TEXT
from .verification import lexical_entailment

SUPPORT_THRESHOLD = 0.65
REJECT_THRESHOLD = 0.4

_ABSTAIN_MARKERS = (
    ABSTAIN_TEXT.lower(),
    "don't have enough evidence",
    "do not have enough evidence",
    "cannot answer",
    "insufficient evidence",
    "i don't know",
)


def is_abstention(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


@dataclass
class AnswerJudgement:
    n_claims: int
    n_unsupported: int
    groundedness: float


def judge_answer(answer: str, corpus_chunks: Sequence[Chunk]) -> AnswerJudgement:
    """Grade an answer claim-by-claim against the whole corpus."""
    claims = split_into_claims_fallback(answer)
    if not claims:
        return AnswerJudgement(0, 0, 1.0)
    corpus_text = "\n".join(c.text for c in corpus_chunks)
    unsupported = 0
    total_score = 0.0
    for claim in claims:
        score = lexical_entailment(claim.text, corpus_text)
        if score < REJECT_THRESHOLD:
            unsupported += 1
            continue
        total_score += 1.0 if score >= SUPPORT_THRESHOLD else 0.5
    return AnswerJudgement(
        n_claims=len(claims),
        n_unsupported=unsupported,
        groundedness=total_score / len(claims),
    )


@dataclass
class QuestionRecord:
    question: str
    qtype: str                      # "answerable" | "unanswerable" | "adversarial"
    gold_keywords: List[str]
    answer: str
    abstained: bool
    latency_s: float = 0.0
    confidence: Optional[float] = None


def keyword_hit(answer: str, gold_keywords: Sequence[str]) -> bool:
    low = answer.lower()
    return any(k.lower() in low for k in gold_keywords)


def evaluate_run(
    records: Sequence[QuestionRecord], corpus_chunks: Sequence[Chunk]
) -> Dict[str, Optional[float]]:
    """Aggregate metrics for one system over the benchmark dataset."""
    total_claims = 0
    unsupported_claims = 0
    groundedness_values: List[float] = []
    hallucinated_questions = 0

    answerable = [r for r in records if r.qtype == "answerable"]
    unanswerable = [r for r in records if r.qtype != "answerable"]

    correct = 0
    answered_answerable = 0
    false_abstentions = 0
    correct_abstentions = 0

    for record in records:
        if record.abstained:
            if record.qtype == "answerable":
                false_abstentions += 1
            else:
                correct_abstentions += 1
            continue
        judgement = judge_answer(record.answer, corpus_chunks)
        total_claims += judgement.n_claims
        unsupported_claims += judgement.n_unsupported
        groundedness_values.append(judgement.groundedness)
        hallucinated = judgement.n_unsupported > 0
        if record.qtype != "answerable":
            # answering an unanswerable question at all is a hallucination
            hallucinated = True
        if hallucinated:
            hallucinated_questions += 1
        if record.qtype == "answerable":
            answered_answerable += 1
            if keyword_hit(record.answer, record.gold_keywords):
                correct += 1

    n = len(records)
    mean_latency = sum(r.latency_s for r in records) / n if n else 0.0
    return {
        "questions": float(n),
        # headline: fraction of questions whose answer contained >=1 fabricated
        # claim (or that answered an unanswerable question)
        "hallucination_rate": hallucinated_questions / n if n else 0.0,
        # claim-level fabrication rate over answered questions
        "unsupported_claim_rate": (
            unsupported_claims / total_claims if total_claims else 0.0
        ),
        "mean_groundedness": (
            sum(groundedness_values) / len(groundedness_values)
            if groundedness_values
            else None
        ),
        "abstention_recall": (
            correct_abstentions / len(unanswerable) if unanswerable else None
        ),
        "false_abstention_rate": (
            false_abstentions / len(answerable) if answerable else None
        ),
        "answer_accuracy": correct / answered_answerable if answered_answerable else None,
        "answer_coverage": (
            answered_answerable / len(answerable) if answerable else None
        ),
        "mean_latency_s": mean_latency,
    }


def record_is_correct(record: "QuestionRecord", corpus_chunks: Sequence[Chunk]) -> bool:
    """Ground-truth correctness for a single answered/abstained question.

    Answerable: answered, hits a gold keyword, and contains no fabricated claim.
    Unanswerable/adversarial: abstaining is the correct behavior.
    """
    if record.qtype == "answerable":
        if record.abstained:
            return False
        if not keyword_hit(record.answer, record.gold_keywords):
            return False
        return judge_answer(record.answer, corpus_chunks).n_unsupported == 0
    return record.abstained


def calibration_report(
    records: Sequence["QuestionRecord"], corpus_chunks: Sequence[Chunk]
) -> Dict[str, Optional[float]]:
    """ECE / AUROC / AURC over records that carry a confidence value."""
    pairs = [
        (r.confidence, record_is_correct(r, corpus_chunks))
        for r in records
        if r.confidence is not None
    ]
    _points, aurc = risk_coverage(pairs)
    return {
        "n_with_confidence": float(len(pairs)),
        "ece": expected_calibration_error(pairs),
        "auroc": auroc(pairs),
        "aurc": aurc,
    }


def expected_calibration_error(
    pairs: Sequence, n_bins: int = 10
) -> Optional[float]:
    """ECE over (confidence, correct) pairs. Lower is better; 0 = perfect.

    Bins predictions by confidence and averages |accuracy - confidence| per bin,
    weighted by bin population.
    """
    pairs = [(c, bool(y)) for c, y in pairs if c is not None]
    if not pairs:
        return None
    n = len(pairs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        # last bin is closed on the right so confidence == 1.0 is included
        bucket = [
            (c, y) for c, y in pairs
            if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)
        ]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, y in bucket if y) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - avg_conf)
    return ece


def auroc(pairs: Sequence) -> Optional[float]:
    """AUROC of confidence as a discriminator of correctness.

    Computed via the Mann-Whitney U statistic (probability a correct answer is
    ranked above an incorrect one). 0.5 = no discrimination, 1.0 = perfect.
    """
    scored = [(c, bool(y)) for c, y in pairs if c is not None]
    pos = [c for c, y in scored if y]
    neg = [c for c, y in scored if not y]
    if not pos or not neg:
        return None
    # sum over pairs of 1[pos>neg] + 0.5*1[pos==neg]
    greater = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                greater += 1
            elif p == q:
                ties += 1
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


def risk_coverage(pairs: Sequence):
    """Risk–coverage curve for selective prediction.

    Sort predictions by confidence (desc); at each coverage level return the
    error rate ("risk") among the most-confident answers retained. Returns a
    list of ``(coverage, risk)`` points plus the area under the curve (AURC;
    lower is better).
    """
    scored = sorted(
        [(c, bool(y)) for c, y in pairs if c is not None],
        key=lambda t: t[0],
        reverse=True,
    )
    if not scored:
        return [], None
    points = []
    errors = 0
    for i, (_, correct) in enumerate(scored, start=1):
        if not correct:
            errors += 1
        points.append((i / len(scored), errors / i))
    aurc = sum(r for _, r in points) / len(points)
    return points, aurc


def citation_precision(
    verdict_records: Sequence, chunks_by_id: Dict[str, Chunk]
) -> Optional[float]:
    """Fraction of citations whose cited chunk actually supports the claim.

    ``verdict_records`` is a flat list of ``ClaimVerdict``; VERITAS-only since
    the baseline emits no citations.
    """
    checked = 0
    good = 0
    for verdict in verdict_records:
        for cid in verdict.claim.citations:
            chunk = chunks_by_id.get(cid)
            if chunk is None:
                checked += 1  # citation to a nonexistent chunk is wrong
                continue
            checked += 1
            if lexical_entailment(verdict.claim.text, chunk.text) >= REJECT_THRESHOLD:
                good += 1
    if checked == 0:
        return None
    return good / checked
