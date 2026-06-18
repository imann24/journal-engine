# ANALYSIS.md — the signal pipeline

How the journal engine turns raw entries into the analytics you see in the
dashboard. The guiding idea: **derivation is decoupled from presentation.** Passes
write typed, versioned signals to a store; everything downstream is a query.

## The five layers

```
 1. corpus        entries table (LanceDB): one row per chunk, with text, date,
                  word_count, bge-m3 vector. Ingest is idempotent (see ingest.py).
       │
       ▼
 2. derivation    passes/deterministic.py — a Pass turns Chunks into SignalRows.
                  LexicalPass (psycholinguistic rates) + GoEmotionsPass (28-label
                  emotion vector). Deterministic by construction: fixed weights /
                  lexicon lookups, no sampling.
       │
       ▼
 3. signal store  journal/signal_store.py — LanceDB `signals` table, one row per
                  (chunk_id, namespace, key). PassRunner asks the store has(...)
                  and only computes new/changed chunks. Idempotent + versioned.
       │
       ▼
 4. analysis      chunk -> entry rollups materialized into `entry_signals`
                  (aggregate_entries); entry -> period (day/week/month/year)
                  computed on read. Future: change-point detection + theme
                  discovery (journal/analysis/, interfaces only).
       │
       ▼
 5. presentation  app.py Emotion + Introspection tabs — pure reads of
                  entry_signals via emotion_entry_wide / lexical_entry_wide.
```

Adding a signal no longer touches the corpus schema or the dashboard's storage:
write a pass, run `derive`, query the store.

## Tables

`signals` — one row per `(chunk_id, namespace, key)` (the natural key):

| column | meaning |
| --- | --- |
| `chunk_id`, `entry_id` | identity (chunk id is the entries table's `id`) |
| `namespace` | `emotion` \| `lexical` (a future LLM pass gets its own) |
| `key` | `goemotions`, `self_focus`, `temporal_focus`, … |
| `value_num` | numeric signals (nullable) |
| `value_json` | structured signals: emotion vector, temporal dict (JSON string, nullable) |
| `model_tag` | reproducibility fingerprint (revision / lexicon hash) |
| `content_hash` | hash of the **chunk text** (the passes' own 16-char hash) |
| `computed_at` | ISO timestamp |
| `pass_name`, `pass_version` | bookkeeping the runner's `has()` is keyed on |

`entry_signals` — the chunk→entry rollup (rebuilt wholesale from `signals`), one
row per `(entry_id, namespace, key)` plus `date`/`date_int` for cheap temporal
grouping. Emotion rolls up as length-weighted **mean** + per-label **peak**
(`goemotions_mean` / `goemotions_peak`); lexical rates roll up as token-weighted
means; `temporal_focus` rolls up per sub-key.

## Idempotency & reproducibility

- **Skip-or-recompute** is decided in `PassRunner` (not the passes) from
  `(pass_name, pass_version, chunk content_hash)`. Edit a chunk → its hash changes
  → only it recomputes. Bump a pass `version` → that pass recomputes everywhere.
- **GoEmotions is pinned to a commit hash** (`config.GOEMOTIONS_REVISION`), so a
  re-pull of the model can never silently shift the numbers.
- **Lexicons are hashed into the lexical `model_tag`**, so any lexicon edit is a
  new fingerprint; bump `LexicalPass.version` to trigger a clean re-derive.
- **`derive --rebuild`** forces recompute (ignores `has()`) without bumping
  versions — for re-deriving after a non-versioned environment change.

## Running it

```bash
python cli.py derive --pass lexical            # rates over every chunk
python cli.py derive --pass goemotions         # 28-label emotion vectors (GPU)
python cli.py derive --pass lexical --since 2024-01-01   # only recent chunks
python cli.py derive --aggregate               # rebuild entry_signals only
```

Each `derive --pass …` run also rebuilds `entry_signals`. Re-running a pass with
no new/changed chunks writes **0** rows. The Emotion/Introspection dashboard tabs
gate behind `config.MIN_SIGNAL_ENTRIES` and surface per-period entry counts so a
sparse history never looks more precise than it is.

## Adding a new pass

1. Implement the `Pass` protocol (`name`, `version`, `model_tag`, `compute`),
   returning `SignalRow`s under a new `namespace`. Make `model_tag` capture
   everything that affects the output (model revision, lexicon/schema hash).
2. Run it: `LanceSignalStore().run_pass(MyPass(), iter_chunks())` (or wire a
   `--pass` choice in `cli.py`).
3. If it needs an entry-level rollup, extend `aggregate_entries`.
4. Query it in the dashboard with a small `*_entry_wide` helper.

### Reserved extension points (not built)
- `passes/llm_profile.py` — `EmotionalProfile`, a future local Nemotron +
  Instructor pass. Same `Pass` contract; **own namespace** so non-deterministic
  LLM signals never mix with the deterministic ones.
- `journal/analysis/` — interfaces/TODOs for change-point detection (`ruptures`)
  over signal series and theme discovery (UMAP + HDBSCAN) over the existing
  `bge-m3` vectors. No dependencies added yet (needs approval per CLAUDE.md).
