"""Adapter tests with fake transports — no network, no SDK installs needed."""

from types import SimpleNamespace

from veritas.llm import (
    AnthropicClient,
    HFInferenceClient,
    MockLLM,
    OpenAICompatClient,
)
from veritas.prompts import (
    decompose_prompt,
    grounded_answer_prompt,
    repair_prompt,
    verify_prompt,
)


class FakeAnthropic:
    def __init__(self):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="anthropic says hi")]
        )


class FakeOpenAI:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai says hi"))]
        )


class FakeHF:
    def __init__(self):
        self.kwargs = None

    def chat_completion(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hf says hi"))]
        )


def test_anthropic_adapter_shapes_request_and_drops_temperature():
    fake = FakeAnthropic()
    client = AnthropicClient(model="claude-opus-4-8", client=fake)
    out = client.complete("hello", system="sys", temperature=0.7, max_tokens=99)
    assert out == "anthropic says hi"
    assert fake.kwargs["model"] == "claude-opus-4-8"
    assert fake.kwargs["system"] == "sys"
    assert fake.kwargs["max_tokens"] == 99
    assert fake.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    # claude-opus-4-8 rejects sampling params: adapter must not send temperature
    assert "temperature" not in fake.kwargs


def test_anthropic_adapter_keeps_temperature_for_older_models():
    fake = FakeAnthropic()
    client = AnthropicClient(model="claude-haiku-4-5", client=fake)
    client.complete("hello", temperature=0.2)
    assert fake.kwargs["temperature"] == 0.2
    assert "system" not in fake.kwargs


def test_openai_adapter_shapes_request():
    fake = FakeOpenAI()
    client = OpenAICompatClient(model="gpt-4o-mini", client=fake)
    out = client.complete("hello", system="sys", temperature=0.3, max_tokens=42)
    assert out == "openai says hi"
    assert fake.kwargs["model"] == "gpt-4o-mini"
    assert fake.kwargs["temperature"] == 0.3
    assert fake.kwargs["max_tokens"] == 42
    assert fake.kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert fake.kwargs["messages"][1] == {"role": "user", "content": "hello"}


def test_hf_adapter_clamps_zero_temperature():
    fake = FakeHF()
    client = HFInferenceClient(model="some/model", client=fake)
    out = client.complete("hello", temperature=0.0)
    assert out == "hf says hi"
    assert fake.kwargs["temperature"] == 0.01


# ---------------------------------------------------------------- MockLLM


def _retrieved(retriever):
    return retriever.retrieve("Mount Everest height meters", k=2)


def test_mock_dispatches_on_task_markers(retriever):
    llm = MockLLM()
    retrieved = _retrieved(retriever)
    evidence_text = "\n".join(f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in retrieved)

    llm.complete(grounded_answer_prompt("How tall is Everest?", retrieved))
    llm.complete(decompose_prompt("A fact. Another fact."))
    llm.complete(verify_prompt("Everest is the highest mountain.", evidence_text))
    llm.complete(repair_prompt("Everest is 9999 meters.", evidence_text))
    assert llm.calls == ["generate", "decompose", "verify", "repair"]

    assert llm.complete("random text") == "UNRECOGNIZED TASK"


def test_mock_verify_outputs_parseable_verdict(retriever):
    llm = MockLLM()
    retrieved = _retrieved(retriever)
    evidence_text = "\n".join(f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in retrieved)
    raw = llm.complete(verify_prompt("Mount Everest is the highest mountain on Earth.", evidence_text))
    assert "VERDICT: SUPPORTED" in raw
    raw = llm.complete(verify_prompt("Cheese is made on the Moon by robots.", evidence_text))
    assert "VERDICT: UNSUPPORTED" in raw


def test_mock_repair_returns_remove_when_no_evidence(retriever):
    llm = MockLLM()
    retrieved = _retrieved(retriever)
    evidence_text = "\n".join(f"[{sc.chunk.chunk_id}] {sc.chunk.text}" for sc in retrieved)
    raw = llm.complete(repair_prompt("Bananas are rich in potassium.", evidence_text))
    assert raw == "REMOVE"


def test_mock_hallucination_injection_is_deterministic(retriever):
    retrieved = _retrieved(retriever)
    prompt = grounded_answer_prompt("What is the height of the summit of Mount Everest in meters?", retrieved)
    a = MockLLM(hallucination_rate=1.0, seed=5).complete(prompt)
    b = MockLLM(hallucination_rate=1.0, seed=5).complete(prompt)
    clean = MockLLM(hallucination_rate=0.0, seed=5).complete(prompt)
    assert a == b
    assert a != clean
