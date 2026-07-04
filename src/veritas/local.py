"""Local open-weights chat model as an ``LLMClient`` — for free live testing on
a Colab / local GPU instead of a paid Inference API.

Unlike ``techniques.decoding.LocalHFClient`` (which drives raw ``generate`` for
DoLa layer-contrasting on *base* models), this client applies the model's
**chat template**, so instruction-tuned open models (Qwen2.5-Instruct,
Llama-3.x-Instruct, Mistral-Instruct, ...) behave correctly. With
``load_4bit=True`` a 7-8B model fits comfortably in a free T4's ~15 GB.

    from veritas.local import LocalChatClient
    llm = LocalChatClient("Qwen/Qwen2.5-7B-Instruct", load_4bit=True)
    # then use `llm` with any technique, exactly like MockLLM / AnthropicClient

Lazily imports ``torch``/``transformers`` — nothing here is needed offline.
"""

from __future__ import annotations

from typing import Optional


class LocalChatClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        load_4bit: bool = False,
        device: Optional[str] = None,
        _model=None,
        _tokenizer=None,
    ):
        self.model_name = model
        if _model is not None and _tokenizer is not None:  # test injection
            self.model, self.tokenizer = _model, _tokenizer
            self.device = device or "cpu"
            return
        try:  # pragma: no cover - requires torch + transformers + a GPU/CPU model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "pip install 'veritas-rag[local]' (torch + transformers) to run "
                "a local model"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        kwargs = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            kwargs["device_map"] = "auto"  # bitsandbytes places layers itself
        else:
            kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not load_4bit:
            self.model = self.model.to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=384):
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
