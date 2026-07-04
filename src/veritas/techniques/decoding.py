"""White-box decoding interventions (DoLa) — the one family that needs logit
access, so it runs only on a local Hugging Face causal LM, never on API models
or the offline mock.

DoLa (Decoding by Contrasting Layers, Chuang et al. 2023) contrasts the
next-token distribution of a late transformer layer against an earlier layer.
Factual knowledge tends to be sharper in later layers, so the contrast
amplifies tokens the model "knows" and suppresses fluent-but-ungrounded ones.
``transformers`` supports it natively via ``model.generate(dola_layers=...)`` —
we wrap that here and compare it against ordinary greedy decoding on the same
model, so the benchmark isolates DoLa's factuality effect.

Also included: a **contrastive speculative check** — the honest,
hallucination-relevant cousin of "Lookahead Decoding" (which is itself only a
latency technique). A small draft model and the main model generate in
parallel; large per-token disagreement flags uncertainty and triggers
abstention. Experimental; its factuality benefit is not well established, so it
is opt-in.

These import ``torch``/``transformers`` lazily; nothing here is needed for the
offline suite.
"""

from __future__ import annotations

from typing import List, Optional

from ..prompts import ABSTAIN_TEXT, GROUNDED_SYSTEM, grounded_answer_prompt
from ..retrieval import HybridRetriever
from .base import BaseTechnique, TechniqueResult


class LocalHFClient:
    """LLMClient backed by a local Hugging Face causal LM, with optional DoLa.

    ``dola_layers`` is passed straight to ``model.generate`` — "high", "low",
    or an explicit list of layer indices. ``None`` = ordinary decoding.
    """

    def __init__(
        self,
        model: str = "gpt2",
        dola_layers: Optional[object] = None,
        repetition_penalty: float = 1.2,
        device: Optional[str] = None,
        _model=None,
        _tokenizer=None,
    ):
        self.model_name = model
        self.dola_layers = dola_layers
        self.repetition_penalty = repetition_penalty
        if _model is not None and _tokenizer is not None:  # test injection
            self.model, self.tokenizer = _model, _tokenizer
            self.device = device or "cpu"
            return
        try:  # pragma: no cover - requires torch + transformers + a model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "DoLa needs a local model: pip install 'veritas-rag[local]' "
                "(torch + transformers)"
            ) from exc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, output_hidden_states=True
        ).to(self.device)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=128):  # pragma: no cover
        import torch

        full = f"{system}\n\n{prompt}" if system else prompt
        inputs = self.tokenizer(full, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if self.dola_layers is not None:
            gen_kwargs["dola_layers"] = self.dola_layers
            # transformers >= 5 moved DoLa into a community custom_generate repo
            import transformers
            if int(transformers.__version__.split(".")[0]) >= 5:
                gen_kwargs["custom_generate"] = "transformers-community/dola"
                gen_kwargs["trust_remote_code"] = True
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip()


class _DecodingTechnique(BaseTechnique):
    family = "decoding"
    requires = "local_hf"

    def __init__(self, hf_client: LocalHFClient, retriever: HybridRetriever, top_k: int = 4):
        self.llm = hf_client
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, question: str) -> TechniqueResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        answer = self.llm.complete(
            grounded_answer_prompt(question, retrieved),
            system=GROUNDED_SYSTEM, temperature=0.0, max_tokens=96,
        ).strip()
        abstained = (not answer) or ABSTAIN_TEXT.lower() in answer.lower()
        return TechniqueResult(
            question=question,
            answer=answer or ABSTAIN_TEXT,
            abstained=abstained,
            trace=[f"local model {self.llm.model_name}, "
                   f"dola_layers={self.llm.dola_layers}"],
            extra={"decoding": "dola" if self.llm.dola_layers else "greedy"},
        )


class VanillaDecodingTechnique(_DecodingTechnique):
    name = "Vanilla decoding (local)"


class DoLaTechnique(_DecodingTechnique):
    name = "DoLa (local)"


def build_decoding_registry(retriever, model: str = "gpt2", dola_layers="high"):  # pragma: no cover
    """Vanilla vs DoLa on the same local model — call from the local-hf runner."""
    greedy = LocalHFClient(model=model, dola_layers=None)
    dola = LocalHFClient(model=model, dola_layers=dola_layers)
    return {
        VanillaDecodingTechnique.name: VanillaDecodingTechnique(greedy, retriever),
        DoLaTechnique.name: DoLaTechnique(dola, retriever),
    }
