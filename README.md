# Journal Engine

A fully local ingestion / query / analysis system for ~12 years of plain-text
journal entries, with a web UI. Runs entirely on the DGX Spark — LanceDB for
storage + full-text search, Ollama (localhost) for embeddings and generation.
**No cloud, no telemetry, ever.** You reach the UI from your Mac over Tailscale.

## What it does
- **Ingest** `.txt` entries from a directory, a drop folder, pasted text, or
  uploads — idempotent and incremental (dedup by content hash; re-running only
  processes new/changed entries).
- **Date** each entry by **filename → header line → file mtime** (or an explicit
  date for pastes), recording which source was used and warning when too many fall
  back to mtime.
- **Search** with hybrid dense (`bge-m3`) + BM25 retrieval fused by reciprocal-rank
  fusion, with a date-range prefilter.
- **Ask** questions (RAG) with answers that cite entry dates and refuse gracefully
  when the excerpts don't cover the question.
- **Enrich** entries with mood (1–5), topics, people, and places via a re-runnable
  LLM pass, then explore them on an **analysis dashboard**.

## Prerequisites
- Python 3.10+ on the Spark.
- Ollama running locally with the models pulled:
  ```bash
  ollama pull bge-m3                  # embeddings (1024-dim)
  ollama pull nemotron-3-super:latest # generation/enrichment (or your tag)
  ```

## Setup
```bash
cd journal-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set a strong JOURNAL_PASSWORD. .env is gitignored — never commit it.
```

## Run the web UI (reach it from your Mac over Tailscale)
```bash
./run_web.sh
```
It binds to `JOURNAL_WEB_HOST` (default `0.0.0.0`) on `JOURNAL_WEB_PORT`
(default `8501`) and prints the exact URL. On this Spark that is:

> **http://spark-0d62.tail9e6b2f.ts.net:8501**

Open that from the Mac Studio. The first screen asks for `JOURNAL_PASSWORD`; once
unlocked, the session stays authenticated until you press **Log out**.

**Keep it tailnet-only.** Binding `0.0.0.0` exposes it on every local interface,
but it should only ever be reachable over your tailnet. Do **not** port-forward it
or open the port on the Spark's WAN firewall. (To scope it strictly to Tailscale,
set `JOURNAL_WEB_HOST` to the Spark's Tailscale IP, e.g. `100.81.153.68`.)

### Web UI areas
1. **Add entries** — paste a single entry (optional date; inferred if blank), paste
   a batch (split on `---`/`===` or blank lines), or upload multiple `.txt` files.
   Submitting dates → chunks → embeds → indexes immediately and shows you exactly
   what was ingested and the date source used for each.
2. **Analysis** — charts for entries/year, mean mood/year, and top
   people/places/topics, with a date-range filter and a button to run/refresh
   enrichment.
3. **Query** — chat-style RAG with optional From/To dates, showing the answer,
   cited entry dates, and the underlying excerpts.

## CLI
```bash
python cli.py ingest ./journals                 # batch ingest a directory
python cli.py ingest                            # sweep the drop folder
python cli.py watch                             # auto-ingest the drop folder on change
python cli.py enrich [--limit N]                # LLM tagging pass (resumable)
python cli.py search "panic about money" --from 2019-01-01 --to 2019-12-31
python cli.py ask "how did I talk about Max over time?"
python cli.py stats
```

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `JOURNAL_PASSWORD` | *(required)* | Web UI password. App refuses to start if unset. |
| `JOURNAL_WEB_HOST` | `0.0.0.0` | Bind address. |
| `JOURNAL_WEB_PORT` | `8501` | Bind port. |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama endpoint. |
| `JOURNAL_EMBED_MODEL` | `bge-m3` | Embedding model. |
| `JOURNAL_EMBED_DIM` | `1024` | Embedding dimension. |
| `JOURNAL_CHAT_MODEL` | `nemotron-3-super:latest` | Generation/enrichment model. Use `nemotron-3-nano:latest` for faster bulk enrichment. |
| `JOURNAL_DB` | `./journal_lancedb` | LanceDB path. |
| `JOURNAL_DROP_DIR` | `./drop` | Watched drop folder. |
| `JOURNAL_CHUNK_WORDS` | `400` | Split entries longer than this. |
| `JOURNAL_CHUNK_OVERLAP` | `1` | Paragraphs of overlap between sub-chunks. |

See `.env.example` for the full list.

## Privacy & security
- `JOURNAL_PASSWORD` lives only in the gitignored `.env` (never hardcoded, never
  committed) and is checked with `hmac.compare_digest`.
- `.gitignore` excludes your entries (`*.txt`, `journals/`, `drop/`), the index
  (`journal_lancedb/`), and `.env`. Your corpus, embeddings, and password never
  enter git.
- Streamlit telemetry is disabled. Nothing leaves the machine.

## Tests
```bash
python -m pytest        # no Ollama required
```
Focus is `tests/test_dating.py` (date inference — the part most likely to be
silently wrong), plus chunking and ingest-idempotency tests.

## Decisions
See [DECISIONS.md](DECISIONS.md) for choices made during the build.
