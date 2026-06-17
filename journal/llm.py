"""Generation via local Ollama (configurable model tag, defaults to Nemotron)."""

from __future__ import annotations

import ollama

from . import config


def chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=config.CHAT_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()
