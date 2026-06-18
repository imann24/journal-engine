"""Extension point for an LLM-based ``EmotionalProfile`` pass — STUB, not wired.

The signal store treats every derivation identically: anything implementing the
``Pass`` protocol (see ``passes/deterministic.py``) can write ``SignalRow``s and be
run by ``PassRunner`` through ``journal.signal_store.LanceSignalStore``. This file
reserves a clean slot for a *non-deterministic* pass: a local Nemotron model
(via Ollama) coaxed into structured output with Instructor, producing a richer
per-entry emotional/cognitive profile than the deterministic passes can.

Intentionally NOT implemented. Notes for whoever builds it:

* Keep it **local**: generation goes through Ollama on localhost (see
  ``journal.llm.chat`` / ``journal.config.CHAT_MODEL``). No cloud APIs, ever.
* LLM output is **not deterministic**. Store these signals under their own
  namespace (e.g. ``"profile"``) so they never mix with the deterministic
  ``emotion``/``lexical`` signals, and fold the model tag + decoding params
  (temperature, seed, schema hash) into ``model_tag`` so a prompt/model change
  re-derives cleanly — exactly as the lexicon hash does today.
* ``version`` bumps on any prompt/schema change. The store already re-derives
  when (pass_name, version, content_hash) changes; nothing else is needed.
* ``Instructor`` is an extra dependency — get approval before adding it (see
  the approved-dependency list in CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from .deterministic import Chunk, SignalRow


class EmotionalProfile:
    """Future LLM pass. Implements the ``Pass`` protocol but does no work yet."""

    name = "emotional_profile"
    version = "v0"

    def __init__(self, model: str | None = None):
        # model defaults to config.CHAT_MODEL when implemented.
        self._model = model

    @property
    def model_tag(self) -> str:
        return f"{self.name}:{self.version}:{self._model or 'default'}"

    def compute(self, chunks: Sequence[Chunk]) -> list[SignalRow]:
        raise NotImplementedError(
            "EmotionalProfile is a reserved extension point. Implement a local "
            "Nemotron + Instructor pass writing SignalRows under a 'profile' "
            "namespace; see this module's docstring."
        )
