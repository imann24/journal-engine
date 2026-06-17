# Decisions

Reasonable choices made during the one-shot build, so they're easy to revisit.

1. **Streamlit over Gradio.** Both were acceptable; Streamlit's `session_state`
   gives a clean native password gate and the dashboard/chat widgets needed, with
   no extra auth dependency.

2. **Default chat model = `nemotron-3-super:latest`.** You run Nemotron; this is
   the flagship you have pulled and it fits the Spark's 128 GB unified memory. It's
   slow for bulk enrichment over 12 years of entries, so the README/`.env` call out
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

8. **Bind `0.0.0.0` by default**, as requested, for Tailscale reachability. The
   README documents keeping it tailnet-only and the stricter option of binding the
   Spark's Tailscale IP directly.

9. **Minimal hand-rolled `.env` loader** in `config.py` instead of `python-dotenv`
   (not on the approved list). It never overrides already-set environment
   variables.

10. **Batch-paste splitting** on a line of `---`/`===` or two-plus blank lines —
    a simple, predictable convention documented in the UI.
