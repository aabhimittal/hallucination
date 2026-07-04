"""DoLa / decoding techniques.

The wiring is tested with a stub client (offline). A real gpt2+DoLa smoke test
runs only when torch + transformers are installed.
"""

import importlib.util

import pytest

from veritas.techniques.decoding import (
    DoLaTechnique,
    VanillaDecodingTechnique,
    build_decoding_registry,
)


class StubHFClient:
    """Duck-typed LocalHFClient: records the decoding config, echoes evidence."""

    def __init__(self, model_name="gpt2", dola_layers=None):
        self.model_name = model_name
        self.dola_layers = dola_layers
        self.last_temperature = None

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=128):
        self.last_temperature = temperature
        # pretend the model extracted the first evidence line
        for line in prompt.splitlines():
            if line.startswith("[c"):
                return line.split("] ", 1)[-1]
        return ""


def test_decoding_techniques_require_local_hf(retriever):
    vanilla = VanillaDecodingTechnique(StubHFClient(dola_layers=None), retriever)
    dola = DoLaTechnique(DoLaClient := StubHFClient(dola_layers="high"), retriever)
    assert vanilla.requires == "local_hf" and dola.requires == "local_hf"
    assert vanilla.family == "decoding"


def test_decoding_technique_runs_at_temperature_zero(retriever):
    client = StubHFClient(dola_layers="high")
    tech = DoLaTechnique(client, retriever)
    result = tech.answer("How tall is Mount Everest?")
    assert result.extra["decoding"] == "dola"
    assert client.last_temperature == 0.0
    assert "dola_layers=high" in result.trace[0]


def test_vanilla_marks_greedy(retriever):
    tech = VanillaDecodingTechnique(StubHFClient(dola_layers=None), retriever)
    result = tech.answer("How tall is Mount Everest?")
    assert result.extra["decoding"] == "greedy"


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("torch") is None,
    reason="requires torch + transformers (local-hf runtime)",
)
def test_dola_real_model_smoke(retriever):  # pragma: no cover - heavy, opt-in
    reg = build_decoding_registry(retriever, model="sshleifer/tiny-gpt2",
                                  dola_layers="high")
    assert set(reg) == {"Vanilla decoding (local)", "DoLa (local)"}
    result = reg["DoLa (local)"].answer("How tall is Mount Everest?")
    assert isinstance(result.answer, str)
