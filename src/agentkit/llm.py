"""LLM backends behind one tiny interface: `chat(messages) -> str`."""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    name: str

    def chat(self, messages: list[Message]) -> str: ...


class HFLocalLLM:
    """Open-weights chat model run locally with transformers (CPU is fine for <=2B models)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name.split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def chat(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        with self._torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 512):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("AGENT_MODEL", "claude-haiku-5-5")
        self.name, self.max_tokens = self.model, max_tokens

    def chat(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=rest, temperature=0)
        return r.content[0].text


class ScriptedLLM:
    """Replays canned responses — used for deterministic tests."""

    def __init__(self, responses: Iterable[str], name: str = "scripted"):
        self.responses, self.name = list(responses), name
        self.calls: list[list[Message]] = []

    def chat(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0) if self.responses else '{"final_answer": "I don\'t know"}'
