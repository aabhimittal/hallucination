"""Provider-agnostic LLM layer.

``LLMClient`` is the only interface the pipeline knows about, so VERITAS runs
on top of *any* LLM. Shipped adapters:

- :class:`AnthropicClient` — Claude via the official ``anthropic`` SDK
- :class:`OpenAICompatClient` — OpenAI or any OpenAI-compatible endpoint
- :class:`HFInferenceClient` — Hugging Face Inference API
- :class:`MockLLM` — deterministic, offline, rule-based model used by the
  test suite, the reproducible benchmark, and the keyless demo mode. It can
  inject hallucinations at a configurable rate to simulate an unreliable base
  model, which is what lets the benchmark *measure* how much of that noise the
  VERITAS pipeline removes.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Protocol, Sequence, Tuple

from .chunking import split_cited_sentences
from .prompts import ABSTAIN_TEXT
from .retrieval import content_tokens, tokenize


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        ...


# --------------------------------------------------------------------------
# Real-provider adapters
# --------------------------------------------------------------------------

# Model families that reject sampling parameters entirely; the adapter drops
# temperature for these and relies on prompting alone.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos",
)


class AnthropicClient:
    """Adapter for Claude models via the official ``anthropic`` SDK."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None, client=None):
        self.model = model
        if client is not None:
            self._client = client
        else:  # pragma: no cover - exercised only with a live key
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "pip install 'veritas-rag[anthropic]' to use AnthropicClient"
                ) from exc
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if not self.model.startswith(_NO_SAMPLING_PREFIXES):
            kwargs["temperature"] = temperature
        response = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()


class OpenAICompatClient:
    """Adapter for OpenAI or any OpenAI-compatible endpoint (via ``openai``)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client=None,
    ):
        self.model = model
        if client is not None:
            self._client = client
        else:  # pragma: no cover - exercised only with a live key
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "pip install 'veritas-rag[openai]' to use OpenAICompatClient"
                ) from exc
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class HFInferenceClient:
    """Adapter for the Hugging Face Inference API chat endpoint."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        token: Optional[str] = None,
        client=None,
        max_retries: int = 4,
    ):
        self.model = model
        self.max_retries = max_retries
        if client is not None:
            self._client = client
        else:  # pragma: no cover - exercised only with a live token
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise ImportError(
                    "pip install 'veritas-rag[hf]' to use HFInferenceClient"
                ) from exc
            self._client = InferenceClient(model=model, token=token)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        import time as _time

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    # the HF endpoint rejects temperature == 0 for some backends
                    temperature=max(temperature, 0.01),
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # rate limits / transient 5xx on free tier
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # don't retry auth/credit/bad-request errors (401/402/403/404/400)
                # — only rate limits (429) and server/transient errors are worth it
                if status in (400, 401, 402, 403, 404):
                    raise
                _time.sleep(min(2 ** attempt, 15))
        raise last_exc


# --------------------------------------------------------------------------
# MockLLM — deterministic offline model with hallucination injection
# --------------------------------------------------------------------------

_EVIDENCE_LINE_RE = re.compile(r"^\[(c\d+)\]\s*(.+)$", re.MULTILINE)


def _parse_evidence(prompt: str) -> List[Tuple[str, str]]:
    """Extract ``(chunk_id, text)`` pairs from a prompt's EVIDENCE block."""
    return _EVIDENCE_LINE_RE.findall(prompt)


def _section(prompt: str, header: str) -> str:
    """Return the text after ``header`` up to the next ALL-CAPS section header."""
    idx = prompt.find(header)
    if idx == -1:
        return ""
    rest = prompt[idx + len(header):]
    nxt = re.search(r"\n[A-Z][A-Z _-]{2,}:", rest)
    return (rest[: nxt.start()] if nxt else rest).strip()


def _split_simple_sentences(text: str) -> List[str]:
    return split_cited_sentences(text)


def _overlap(query_terms: Sequence[str], sentence: str) -> float:
    if not query_terms:
        return 0.0
    sent_terms = set(tokenize(sentence))
    return sum(1 for t in query_terms if t in sent_terms) / len(query_terms)


class MockLLM:
    """Deterministic rule-based stand-in for a real LLM.

    Dispatches on the ``TASK:`` marker embedded in every VERITAS prompt.
    With ``hallucination_rate > 0`` it corrupts a fraction of its answers
    (number mutation or fabricated sentences), seeded per-question so runs
    are exactly reproducible.
    """

    def __init__(self, hallucination_rate: float = 0.0, seed: int = 13):
        self.hallucination_rate = hallucination_rate
        self.seed = seed
        self.calls: List[str] = []  # task markers, for test introspection
        self._sample_counter = 0    # advances only on sampled (temp>0) calls

    # ------------------------------------------------------------------ API
    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        # A sampled call (temperature > 0) draws a fresh nonce so repeated
        # identical prompts vary — this is what makes semantic entropy and
        # self-consistency measurable against the mock model.
        if temperature and temperature > 0:
            nonce = self._sample_counter
            self._sample_counter += 1
        else:
            nonce = 0

        if "TASK: GROUNDED_ANSWER" in prompt:
            self.calls.append("generate")
            return self._answer(prompt, grounded=True, nonce=nonce)
        if "TASK: BASELINE_ANSWER" in prompt:
            self.calls.append("baseline")
            return self._answer(prompt, grounded=False, nonce=nonce)
        if "TASK: ANSWER_WITH_CONFIDENCE" in prompt:
            self.calls.append("confidence")
            return self._answer_with_confidence(prompt, nonce=nonce)
        if "TASK: EXTRACT_QUOTES" in prompt:
            self.calls.append("quotes")
            return self._extract_quotes(prompt, nonce=nonce)
        if "TASK: SYNTHESIZE_FROM_QUOTES" in prompt:
            self.calls.append("synthesize")
            return self._synthesize(prompt)
        if "TASK: EDIT_DRAFT" in prompt:
            self.calls.append("edit")
            return _section(prompt, "DRAFT:")  # faithful editor: no new facts
        if "TASK: DECOMPOSE_CLAIMS" in prompt:
            self.calls.append("decompose")
            return self._decompose(prompt)
        if "TASK: VERIFY_CLAIM" in prompt:
            self.calls.append("verify")
            return self._verify(prompt)
        if "TASK: REPAIR_CLAIM" in prompt:
            self.calls.append("repair")
            return self._repair(prompt)
        return "UNRECOGNIZED TASK"

    # ------------------------------------------------------------- answering
    def _rng(self, key: str) -> random.Random:
        return random.Random(f"{self.seed}:{key}")

    def _relevant_sentences(self, question, evidence):
        q_terms = content_tokens(question)
        scored: List[Tuple[float, str, str]] = []
        for chunk_id, text in evidence:
            for sent in _split_simple_sentences(text):
                scored.append((_overlap(q_terms, sent), chunk_id, sent))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(cid, s) for ov, cid, s in scored if ov >= 0.34][:2]

    def _answer(self, prompt: str, grounded: bool, nonce: int = 0) -> str:
        question = _section(prompt, "QUESTION:")
        evidence = _parse_evidence(prompt)
        relevant = self._relevant_sentences(question, evidence)

        # nonce 0 reproduces the original single-call behavior exactly; sampled
        # calls (nonce > 0) vary so multi-sample methods see real spread.
        rng = self._rng(question if nonce == 0 else f"{question}:{nonce}")
        hallucinate = rng.random() < self.hallucination_rate

        if not relevant:
            if grounded:
                return ABSTAIN_TEXT
            # A naively-prompted model guesses instead of abstaining, and each
            # sample guesses differently -> high semantic entropy.
            return self._fabricate(question, rng)

        sentences = [f"{s} [{cid}]" for cid, s in relevant]
        answer = " ".join(sentences)
        if hallucinate:
            answer = self._corrupt(answer, question, evidence, rng)
        return answer

    def _answer_with_confidence(self, prompt: str, nonce: int = 0) -> str:
        question = _section(prompt, "QUESTION:")
        evidence = _parse_evidence(prompt)
        relevant = self._relevant_sentences(question, evidence)
        rng = self._rng(f"conf:{question}:{nonce}")
        if not relevant:
            return f"{ABSTAIN_TEXT}\nCONFIDENCE: 0.10"
        answer = self._answer(prompt.replace("ANSWER_WITH_CONFIDENCE", "GROUNDED_ANSWER"),
                              grounded=True, nonce=nonce)
        # confident when grounded; a fabricated/corrupted answer still reads
        # confident to the model itself -> tests calibration realistically
        conf = round(0.82 + rng.random() * 0.15, 2)
        return f"{answer}\nCONFIDENCE: {conf}"

    def _extract_quotes(self, prompt: str, nonce: int = 0) -> str:
        question = _section(prompt, "QUESTION:")
        evidence = _parse_evidence(prompt)
        relevant = self._relevant_sentences(question, evidence)
        rng = self._rng(f"quote:{question}:{nonce}")
        if not relevant:
            return "QUOTE: NONE"
        lines = [f"QUOTE: {s} [{cid}]" for cid, s in relevant]
        if rng.random() < self.hallucination_rate:
            # a hallucinating model sometimes "quotes" text that isn't there;
            # the grounding verifier must catch it as a non-substring
            lines.append(f"QUOTE: {self._fabricate(question, rng)} [{relevant[0][0]}]")
        return "\n".join(lines)

    def _synthesize(self, prompt: str) -> str:
        # scan the whole prompt for QUOTE: lines (they only appear in the
        # quotes block); _section can't be used because the QUOTE: sub-lines
        # look like section headers to its splitter
        quotes = [
            line.strip()[len("QUOTE:"):].strip()
            for line in prompt.splitlines()
            if line.strip().upper().startswith("QUOTE:")
            and "NONE" not in line.upper()
        ]
        if not quotes:
            return ABSTAIN_TEXT
        return " ".join(quotes)

    @staticmethod
    def _fabricate(question: str, rng: random.Random) -> str:
        subject_terms = content_tokens(question)[:3]
        subject = " ".join(subject_terms) if subject_terms else "this topic"
        year = 1850 + rng.randrange(160)
        qty = 3 + rng.randrange(90)
        template = rng.choice(
            [
                f"The {subject} was first documented in {year} and remains well known today.",
                f"Records indicate that the {subject} involved approximately {qty} distinct cases.",
                f"The {subject} is generally attributed to early pioneers working around {year}.",
            ]
        )
        return template

    def _corrupt(
        self,
        answer: str,
        question: str,
        evidence: List[Tuple[str, str]],
        rng: random.Random,
    ) -> str:
        numbers = re.findall(r"\d[\d,.]*", answer)
        if numbers and rng.random() < 0.6:
            # mutate one number so the claim contradicts the evidence
            target = rng.choice(numbers)
            digits = re.sub(r"[^\d]", "", target) or "1"
            mutated = str(int(digits) + 3 + rng.randrange(40))
            return answer.replace(target, mutated, 1)
        # otherwise append a fabricated sentence miscited to a real chunk
        cid = evidence[0][0] if evidence else "c1"
        extra = self._fabricate(question, rng)
        return f"{answer} {extra} [{cid}]"

    # ----------------------------------------------------------- decompose
    def _decompose(self, prompt: str) -> str:
        answer = _section(prompt, "ANSWER:")
        sentences = _split_simple_sentences(answer)
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    # -------------------------------------------------------------- verify
    def _verify(self, prompt: str) -> str:
        from .verification import lexical_entailment

        claim = _section(prompt, "CLAIM:")
        evidence_text = _section(prompt, "EVIDENCE:")
        score = lexical_entailment(claim, evidence_text)
        if score >= 0.7:
            verdict = "SUPPORTED"
        elif score >= 0.4:
            verdict = "PARTIAL"
        else:
            verdict = "UNSUPPORTED"
        return (
            f"The claim was checked against the evidence sentence by sentence "
            f"(entailment score {score:.2f}).\nVERDICT: {verdict}"
        )

    # -------------------------------------------------------------- repair
    def _repair(self, prompt: str) -> str:
        claim = _section(prompt, "UNSUPPORTED CLAIM:")
        evidence = _parse_evidence(prompt)
        claim_terms = content_tokens(claim)
        best: Tuple[float, str, str] = (0.0, "", "")
        for chunk_id, text in evidence:
            for sent in _split_simple_sentences(text):
                ov = _overlap(claim_terms, sent)
                if ov > best[0]:
                    best = (ov, chunk_id, sent)
        if best[0] >= 0.35:
            return f"{best[2]} [{best[1]}]"
        return "REMOVE"
