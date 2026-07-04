"""LocalChatClient wiring — tested with a fake model/tokenizer (no download).

The real path is exercised on a GPU via the Colab notebook; here we only verify
that the client applies the chat template and slices the generated tokens.
"""

import importlib.util

import pytest

torch = pytest.importorskip("torch")  # skips cleanly when torch isn't installed

from veritas.local import LocalChatClient


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen_messages = messages
        assert add_generation_prompt is True
        return "<chat>" + " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors="pt"):
        # 3 prompt tokens
        return _FakeEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return "8849 meters [c1]"


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, **kwargs):
        # echo prompt tokens + two "generated" tokens
        return torch.cat([input_ids, torch.tensor([[9, 9]])], dim=1)


def test_local_chat_client_applies_chat_template_and_slices():
    tok = _FakeTokenizer()
    client = LocalChatClient(model="fake/model", _model=_FakeModel(), _tokenizer=tok)
    out = client.complete("How tall is Everest?", system="Be grounded.", temperature=0.0)
    assert out == "8849 meters [c1]"
    # system + user messages were passed through the chat template
    assert tok.seen_messages[0] == {"role": "system", "content": "Be grounded."}
    assert tok.seen_messages[1]["role"] == "user"


def test_local_chat_client_is_a_valid_llm_for_techniques(retriever):
    from veritas.techniques import VeritasTechnique

    client = LocalChatClient(model="fake/model", _model=_FakeModel(),
                             _tokenizer=_FakeTokenizer())
    # duck-types as an LLMClient: a technique can call it without error
    result = VeritasTechnique(client, retriever).answer("How tall is Everest?")
    assert isinstance(result.answer, str)
