"""Prompt templates and stage-wise temperature schedule for VERITAS.

Every template embeds an explicit ``TASK:`` marker. That serves two purposes:
it makes intent unambiguous to real LLMs, and it lets the deterministic
``MockLLM`` dispatch on the task without any network access.

The temperature schedule is a core part of the technique: factual grounded
generation runs cold (0.0-0.2), and every verification-side call runs at 0.0
so the checker never introduces new entropy of its own.
"""

from __future__ import annotations

from typing import Sequence

from .retrieval import ScoredChunk

# Stage-wise temperature schedule. Adapters may drop the parameter for models
# that reject sampling params (e.g. Claude Opus 4.7+); the prompts alone still
# carry most of the effect there.
STAGE_TEMPERATURES = {
    "generate": 0.1,
    "decompose": 0.0,
    "verify": 0.0,
    "repair": 0.0,
}

ABSTAIN_TEXT = (
    "I don't have enough evidence in the provided documents to answer this "
    "question reliably."
)

GROUNDED_SYSTEM = (
    "You are a rigorously grounded assistant. You answer ONLY from the "
    "evidence chunks provided. You never use outside knowledge, never guess, "
    "and you say when the evidence is insufficient. Saying \"I don't know\" "
    "is always better than an unsupported answer."
)


def format_evidence(retrieved: Sequence[ScoredChunk]) -> str:
    return "\n".join(f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in retrieved)


def grounded_answer_prompt(question: str, retrieved: Sequence[ScoredChunk]) -> str:
    return f"""TASK: GROUNDED_ANSWER

Answer the question using ONLY the evidence chunks below.

Rules:
1. Every sentence of your answer MUST end with citations of the chunks that
   support it, e.g. "Water boils at 100 C at sea level. [c2]"
2. Use only facts that appear in the evidence. Do not add background
   knowledge, estimates, or plausible-sounding details.
3. If the evidence does not contain the answer, reply exactly:
   "{ABSTAIN_TEXT}"
4. Be concise: 1-4 sentences.

EVIDENCE:
{format_evidence(retrieved)}

QUESTION: {question}

ANSWER:"""


def baseline_answer_prompt(question: str, retrieved: Sequence[ScoredChunk]) -> str:
    """Naive RAG prompt: no citation contract, no abstention permission.

    Used by :class:`veritas.pipeline.BaselineRAG` as the comparison point in
    benchmarks — this is how most quick-start RAG tutorials prompt the model.
    """
    return f"""TASK: BASELINE_ANSWER

Use the context below to answer the question.

CONTEXT:
{format_evidence(retrieved)}

QUESTION: {question}

ANSWER:"""


def decompose_prompt(answer: str) -> str:
    return f"""TASK: DECOMPOSE_CLAIMS

Break the answer below into atomic factual claims. One claim per line,
numbered "1.", "2.", ... Each claim must be a single self-contained factual
statement (resolve pronouns). Keep the citation markers like [c3] attached to
the claim they belong to. Output only the numbered list.

ANSWER:
{answer}

CLAIMS:"""


def verify_prompt(claim: str, evidence_text: str) -> str:
    return f"""TASK: VERIFY_CLAIM

You are verifying one factual claim against evidence, using chain-of-
verification. Think step by step:
1. What does the claim assert (entities, numbers, relations)?
2. Which evidence sentences are relevant?
3. Does the evidence entail the claim, partially support it, or not support it?

Then output your verdict on the final line as exactly one of:
VERDICT: SUPPORTED
VERDICT: PARTIAL
VERDICT: UNSUPPORTED

EVIDENCE:
{evidence_text}

CLAIM: {claim}

ANALYSIS:"""


def quote_extraction_prompt(question: str, retrieved: Sequence[ScoredChunk]) -> str:
    """Long-context quote grounding: pull verbatim quotes before synthesizing."""
    return f"""TASK: EXTRACT_QUOTES

Before answering, extract the exact, word-for-word quotes from the evidence
that are relevant to the question. Copy them verbatim — do not paraphrase,
summarize, or invent. One quote per line, prefixed "QUOTE:" and ending with
its chunk citation. If no evidence is relevant, output "QUOTE: NONE".

EVIDENCE:
{format_evidence(retrieved)}

QUESTION: {question}

QUOTES:"""


def synthesize_from_quotes_prompt(question: str, quotes: str) -> str:
    return f"""TASK: SYNTHESIZE_FROM_QUOTES

Answer the question using ONLY the verbatim quotes below. Every sentence must
cite the quote's chunk. Do not add anything not present in the quotes. If the
quotes do not answer the question, reply exactly: "{ABSTAIN_TEXT}"

QUOTES:
{quotes}

QUESTION: {question}

ANSWER:"""


def answer_with_confidence_prompt(question: str, retrieved: Sequence[ScoredChunk]) -> str:
    """Grounded answer plus a self-reported confidence, for calibration eval."""
    return f"""TASK: ANSWER_WITH_CONFIDENCE

Answer the question using ONLY the evidence. Cite every sentence. Then, on the
final line, report your confidence that the answer is fully supported by the
evidence, as "CONFIDENCE: <number between 0 and 1>". Be honest: report low
confidence when the evidence is thin. If the evidence does not answer the
question, reply "{ABSTAIN_TEXT}" and "CONFIDENCE: 0.1".

EVIDENCE:
{format_evidence(retrieved)}

QUESTION: {question}

ANSWER:"""


def editor_prompt(question: str, draft: str, retrieved: Sequence[ScoredChunk]) -> str:
    """Multi-agent editor pass: tighten the researcher's draft, no new facts."""
    return f"""TASK: EDIT_DRAFT

You are an editor. Improve the draft for clarity and concision WITHOUT adding
any fact not present in the evidence and WITHOUT removing citations. Return
only the edited answer.

EVIDENCE:
{format_evidence(retrieved)}

QUESTION: {question}

DRAFT: {draft}

EDITED:"""


def repair_prompt(claim: str, evidence_text: str) -> str:
    return f"""TASK: REPAIR_CLAIM

The claim below was NOT supported by the evidence. If the evidence contains a
correct statement on the same topic, rewrite the claim so it is fully
supported, ending with the citation of the supporting chunk. If the evidence
contains nothing on this topic, output exactly: REMOVE

EVIDENCE:
{evidence_text}

UNSUPPORTED CLAIM: {claim}

CORRECTED CLAIM:"""
