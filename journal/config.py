"""Central configuration. Everything is overridable via environment variables.

Nothing here is secret. The one secret — JOURNAL_PASSWORD — is read directly in
the web app (app.py) and never stored on disk by this code.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency). Lines are KEY=VALUE; existing
    environment variables are never overridden, and the file is optional.

    JOURNAL_PASSWORD typically lives here — this file is gitignored and must
    never be committed."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env before any config values are read below.
load_dotenv(os.environ.get("JOURNAL_ENV_FILE", ".env"))


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


# --- Storage --------------------------------------------------------------- #
DB_PATH: str = str(_path("JOURNAL_DB", "./journal_lancedb"))
TABLE: str = os.environ.get("JOURNAL_TABLE", "entries")

# Signal store: typed, versioned derivation outputs (see journal/signal_store.py
# and passes/deterministic.py). `signals` holds one row per (chunk, namespace,
# key); `entry_signals` is the chunk->entry rollup materialized for the dashboard.
SIGNALS_TABLE: str = os.environ.get("JOURNAL_SIGNALS_TABLE", "signals")
ENTRY_SIGNALS_TABLE: str = os.environ.get("JOURNAL_ENTRY_SIGNALS_TABLE", "entry_signals")

# Pinned GoEmotions model revision (a commit hash, not "main") so a re-pull can
# never silently move the emotion numbers. Flows into the pass's model_tag.
GOEMOTIONS_REVISION: str = os.environ.get(
    "JOURNAL_GOEMOTIONS_REVISION", "d75048347613a25d77de8cf6412eaae9fa7b26be"
)

# Below this many entries with signals, the Emotion/Introspection dashboard tabs
# refuse to imply precision — they show a "not enough data yet" notice instead.
MIN_SIGNAL_ENTRIES: int = int(os.environ.get("JOURNAL_MIN_SIGNAL_ENTRIES", "30"))

# Folder you drop .txt files into for the `watch` command / manual sweeps.
DROP_DIR: str = str(_path("JOURNAL_DROP_DIR", "./drop"))

# --- Ollama models --------------------------------------------------------- #
# Ollama is local to the Spark. ollama-py honours the OLLAMA_HOST env var; we
# default it to localhost so generation/embeddings never leave the machine.
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("OLLAMA_HOST", OLLAMA_HOST)

EMBED_MODEL: str = os.environ.get("JOURNAL_EMBED_MODEL", "bge-m3")
# bge-m3 dense embeddings are 1024-dim. Override if you swap embedding models.
EMBED_DIM: int = int(os.environ.get("JOURNAL_EMBED_DIM", "1024"))

# Generation / enrichment model. The user runs Nemotron; default to the flagship.
# For bulk enrichment over many years, `nemotron-3-nano:latest` is much faster.
CHAT_MODEL: str = os.environ.get("JOURNAL_CHAT_MODEL", "nemotron-3-super:latest")

# --- Chunking -------------------------------------------------------------- #
CHUNK_TARGET_WORDS: int = int(os.environ.get("JOURNAL_CHUNK_WORDS", "400"))
CHUNK_OVERLAP_PARAS: int = int(os.environ.get("JOURNAL_CHUNK_OVERLAP", "1"))
EMBED_BATCH: int = int(os.environ.get("JOURNAL_EMBED_BATCH", "64"))

# --- Web server ------------------------------------------------------------ #
# Bind to 0.0.0.0 so the Mac can reach it over Tailscale at the Spark's MagicDNS
# name. Keep this tailnet-only: do NOT port-forward or expose it publicly.
WEB_HOST: str = os.environ.get("JOURNAL_WEB_HOST", "0.0.0.0")
WEB_PORT: int = int(os.environ.get("JOURNAL_WEB_PORT", "8501"))

# Fraction of entries falling back to mtime above which we warn loudly.
MTIME_WARN_FRACTION: float = float(os.environ.get("JOURNAL_MTIME_WARN", "0.30"))

# For ambiguous numeric dates like 1.03.25, assume day-first (European) instead
# of the default month-first (US, e.g. 1.03.25 -> Jan 3 2025).
DATE_DAYFIRST: bool = os.environ.get("JOURNAL_DATE_DAYFIRST", "false").lower() in (
    "1", "true", "yes",
)
