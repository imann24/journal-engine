# CLAUDE.md — Journal Engine

Project context and hard constraints for anyone (human or AI) working on this repo.

## What this is
A fully local ingestion / query / analysis engine for ~12 years (2013–present) of
plain-text journal entries, with a Streamlit web UI. This is the most personal
data the owner has. **Everything runs locally on one machine. No cloud APIs, no
telemetry, ever.**

## Non-negotiable constraints
- **Local only.** Embeddings and generation go through Ollama on `localhost`
  (the DGX Spark). Never add a cloud API client. `OLLAMA_HOST` defaults to
  `http://localhost:11434`.
- **No telemetry.** Streamlit usage stats are disabled in `.streamlit/config.toml`
  and via `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` in `run_web.sh`. Don't add
  analytics, crash reporting, or any outbound calls.
- **Never commit personal data or secrets.** `.gitignore` excludes the corpus
  (`*.txt`, `journals/`, `drop/`), the index (`journal_lancedb/`, `*.lance/`), and
  the password file (`.env`). Do not weaken these rules.
- **The web UI is password-gated.** Read `JOURNAL_PASSWORD` from the environment
  (via the gitignored `.env`). Compare with `hmac.compare_digest`. If it is unset,
  the app refuses to start. Login persists per browser via an HMAC-signed cookie
  (`journal/webauth.py` + `extra-streamlit-components` CookieManager) — the cookie
  holds a token, never the password; changing the password invalidates it. There
  is a logout control that clears the cookie.
- **Tailnet-only.** The server binds `0.0.0.0:<port>` so the Mac Studio can reach
  it over Tailscale at the Spark's MagicDNS name. It must never be port-forwarded
  or exposed to the public internet.

## Stack (decided — do not relitigate)
- Storage + full-text search: **LanceDB** (embedded, no server).
- Embeddings: **Ollama `bge-m3`** (1024-dim dense; BM25/FTS is LanceDB's side of
  hybrid search).
- Generation + enrichment: **Ollama**, model tag configurable
  (`JOURNAL_CHAT_MODEL`, default `nemotron-3-super:latest`).
- Retrieval: hybrid dense + BM25 fused with **reciprocal-rank fusion**, with a
  date-range prefilter.
- Web UI: single-process **Streamlit** app (`app.py`).
- Approved dependencies only: `lancedb, ollama, python-dateutil, pandas,
  streamlit, plotly, watchdog, pyarrow, pytest`, plus `extra-streamlit-components`
  (user-approved, for the persistent-login cookie). **Ask before adding any other
  dependency.** (Note: we deliberately read tables via `tbl.to_arrow().to_pandas()`
  to avoid pulling in the optional `pylance` package.)

## Layout
```
journal/
  config.py      env-driven config + a minimal .env loader (no python-dotenv)
  dating.py      date inference (filename -> header -> mtime/today). Heavily tested.
  chunking.py    keep short entries whole; split long ones by paragraph w/ overlap
  embeddings.py  Ollama bge-m3 dense embeddings
  llm.py         Ollama chat
  store.py       LanceDB schema, open/create, dedup bookkeeping; list/remove entries
  webauth.py     pure HMAC token helpers for the persistent-login cookie
  ingest.py      one idempotent/incremental pipeline for all sources
  search.py      hybrid RRF search + date prefilter
  rag.py         grounded Q&A with citations + graceful refusal
  enrich.py      re-runnable per-entry LLM tagging (mood/topics/people/places)
  stats.py       analytics over enriched columns (entry-level)
  watch.py       watchdog drop-folder auto-ingest
cli.py           ingest / enrich / search / ask / stats / watch / list / remove
app.py           Streamlit UI: Add entries / Analysis / Query (all behind auth),
                 sidebar chat-model picker (remembered per browser)
tests/           date inference (critical), chunking, ingest idempotency
run_web.sh       launches the UI bound to JOURNAL_WEB_HOST:JOURNAL_WEB_PORT
```

## Data model (one LanceDB table, one row per chunk)
Dedup/enrichment are keyed to the **entry** (`entry_id` + `content_hash`), not the
chunk. Re-ingesting an unchanged entry is a no-op; an edited entry has its old
chunks deleted and replaced and is re-flagged `enriched = False`. Dates are stored
both as ISO string (`date`) and `int YYYYMMDD` (`date_int`, for cheap range
filters), with the inference source recorded (`date_source`).

## Conventions
- All paths, model tags, the web port, and the password are environment variables
  (see `config.py` / `.env.example`).
- Date inference is the part most likely to be silently wrong — keep
  `tests/test_dating.py` green and extend it when you touch `dating.py`.
- Tests must not require Ollama; embeddings are monkeypatched in ingest tests.

## Run
```bash
python -m pytest                 # tests (no Ollama needed)
python cli.py ingest ./journals  # batch ingest (idempotent)
./run_web.sh                     # web UI, prints the Tailscale URL
```
See README.md for full setup and the exact URL.
