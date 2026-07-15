# Decisions

Reasonable choices made during the one-shot build, so they're easy to revisit.

1. **Streamlit over Gradio.** Both were acceptable; Streamlit's `session_state`
   gives a clean native password gate and the dashboard/chat widgets needed, with
   no extra auth dependency.

2. **Default chat model = `nemotron-3-super:latest`.** Nemotron is a capable local
   model that performs well on hardware with ample VRAM. It can be slow for bulk
   enrichment over large corpora, so the README/`.env` call out
   `nemotron-3-nano:latest` as the faster option — switch via `JOURNAL_CHAT_MODEL`.
   Retrieval/RAG quality is unaffected by the choice.

3. **Read LanceDB via `tbl.to_arrow().to_pandas()`** rather than the built-in
   `tbl.to_pandas()`. On lancedb 0.33 the latter requires the optional `pylance`
   package, which is outside the approved dependency list. Going through Arrow uses
   only `pyarrow` (already approved) and avoids adding a dependency.

4. **Dedup keyed to the entry, not the chunk.** `entry_id` + `content_hash`
   determine new/changed/unchanged. File entries use their relative path as
   `entry_id`; pasted/batch entries use a content-derived id (`paste:<hash>`), so
   identical pastes dedup; uploads use `upload:<filename>`. Changed entries have
   their old chunks deleted and replaced, and are re-flagged for enrichment.

5. **Enrichment runs per entry, written to all its chunks.** Mood/topics/people/
   places describe an entry, so we enrich on the joined entry text (capped for the
   prompt) and write the result to every chunk. An `enriched` flag makes the pass
   resumable and idempotent.

6. **Analytics are entry-level.** Stats collapse chunks back to one row per entry
   so a long, multi-chunk entry counts once.

7. **Explicit Arrow schema with a fixed 1024-dim vector.** Lets the web UI create
   an empty table and ingest the very first entry into a fresh DB. Change
   `JOURNAL_EMBED_DIM` if you swap embedding models.

8. **Bind `0.0.0.0` by default** for Tailscale reachability. The
   README documents keeping it tailnet-only and the stricter option of binding the
   Spark's Tailscale IP directly.

9. **Minimal hand-rolled `.env` loader** in `config.py` instead of `python-dotenv`
   (not on the approved list). It never overrides already-set environment
   variables.

10. **Batch-paste splitting** on a line of `---`/`===` or two-plus blank lines —
    a simple, predictable convention documented in the UI.

11. **Persistent login via a signed cookie** (`extra-streamlit-components`,
    user-approved). The cookie stores an HMAC token (key = `JOURNAL_AUTH_SECRET`,
    defaulting to the password), never the password. Streamlit has no native
    cookie-write API, so a small component is required; this was chosen over a
    URL-token scheme to keep the token out of the URL/history/logs.

12. **Chat-model picker** lists every model from `ollama.list()` (including
    embedding models like bge-m3 — Ollama doesn't reliably distinguish them, and
    the user asked for "any" model). The selection drives RAG + enrichment, is
    remembered per browser in a cookie, and falls back to the default if the
    remembered model is no longer served.

13. **Model choice persisted via the existing cookie store**, not browser
    `localStorage`. localStorage would need an additional component dependency,
    whereas the auth cookie manager is already present; this meets the goal
    (per-browser persistence + availability fallback) with zero new dependencies.

14. **Short / year-last filename dates** (`1.03.25`, `01/03/2025`) are parsed with
    a 1970–2069 two-digit-year pivot and **US month-first** disambiguation by
    default (flip with `JOURNAL_DATE_DAYFIRST`). If month-first is impossible
    (first field > 12) we auto-fall back to day-first.

15. **Entry removal** is exposed both in the CLI (`list`, `remove` by id / date
    range / `--all`, with confirmation unless `--yes`) and the web UI (Manage
    panel). Deletes are by `entry_id` and rebuild the FTS index afterward.

16. **Versioned signal store between corpus and dashboard.** Derivation used to be
    coupled to presentation: every new signal meant a column on the `entries`
    table and a schema change, and a derivation couldn't be re-run in isolation.
    We inserted a typed, versioned **signal store** (`journal/signal_store.py`,
    LanceDB tables `signals` + `entry_signals`) fed by deterministic *passes*
    (`passes/deterministic.py`). Analysis/visualization is now a pure query over
    that store. See `ANALYSIS.md` for the five-layer design.

    - **Natural key** `(chunk_id, namespace, key)`; `upsert` is last-write-wins
      (delete-by-predicate then add, like the entries store's changed-entry path).
    - **Idempotency lives in `PassRunner`**, keyed on `(pass_name, pass_version,
      chunk content_hash)`. A changed chunk or a bumped pass version re-derives;
      an unchanged one is skipped. `cli.py derive --rebuild` forces recompute
      without bumping any version. The signal store reads `has()` from a small
      in-memory set so a full run is one table scan, not one per chunk.
    - `pass_name`/`pass_version` are stored as their own columns (the adapter
      stamps them via `bind_pass`) because neither is recoverable from `model_tag`,
      which encodes only the lexicon hash / model revision.
    - **Reproducibility:** the GoEmotions model is pinned to a commit hash
      (`config.GOEMOTIONS_REVISION`), not `"main"`, so a re-pull can't move the
      numbers; the lexicons are hashed into the lexical pass's `model_tag`, so any
      lexicon edit is a new version and re-derives only that pass.
    - **Deterministic vs LLM signals stay separate.** GoEmotions/lexical are
      deterministic (fixed weights, lexicon lookups — no sampling). A future LLM
      pass (`passes/llm_profile.py`, stubbed) must write under its own namespace
      so non-deterministic signals never mix with these.

17. **spaCy added; transformers/torch optional.** The lexical pass uses spaCy
    (`en_core_web_sm`) for real verb tense and falls back to a marker-word
    approximation when it's absent. `transformers`+`torch` are an *optional* GPU
    extra needed only by the GoEmotions pass (lazy-imported), so the rest of the
    engine — and the test suite — runs without them. GoEmotions runs on GPU
    (`device=0`, fp16) when CUDA is available.

18. **Insight overhaul stays inside the approved stack.** The Home/Explore tabs
    (`journal/insights.py`, `journal/themes.py`, `journal/digest.py`) add no
    dependencies: insights are pure pandas; theme discovery is a small seeded
    k-means written against numpy, which is already a hard transitive dependency
    of pandas/pyarrow (nothing new is installed); digests use the existing
    Ollama `chat`. Clustering is deterministic (seeded k-means++), labeled by
    distinctive words vs. the corpus, so themes are reproducible with no model
    call — LLM naming is an optional, clearly-separated extra.

19. **Period digests are cached in SQLite next to the conversations DB** (inside
    the gitignored LanceDB folder). The cache key hashes the period, the model
    tag, and every in-range entry's `content_hash`, so editing or re-ingesting
    any entry in the range invalidates its digests automatically — no manual
    cache management, and a digest can never silently describe stale text.

20. **RAG became conversational without a query-rewrite model call.** `ask()`
    now passes recent chat history to the model and, for short follow-ups
    (≤5 words), concatenates the previous user question into the retrieval
    query — cheap and predictable versus an extra LLM round-trip for query
    condensation. Date filters can be inferred from the question, but only
    conservatively: explicit years and "last/this year", never guessed seasons
    or months, and an explicit From/To always wins. The applied range is shown
    under the answer so auto-filtering is never invisible.
