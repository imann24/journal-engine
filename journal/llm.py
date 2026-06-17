"""Generation via local Ollama (configurable model tag, defaults to Nemotron)."""

from __future__ import annotations

import ollama

from . import config


def chat(prompt: str, system: str | None = None, temperature: float = 0.2,
         model: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model or config.CHAT_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


def list_models() -> list[str]:
    """Names of every model available on the local Ollama server (for the UI
    model picker). Returns [] if Ollama can't be reached."""
    try:
        resp = ollama.list()
    except Exception:
        return []
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names: list[str] = []
    for m in models or []:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return sorted(set(names))
