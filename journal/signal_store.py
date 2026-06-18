"""LanceDB-backed signal store + chunk loader + chunk->entry aggregation.

This is the layer between the corpus and the dashboard. Deterministic derivation
passes (see ``passes/deterministic.py``) emit typed ``SignalRow``s; this module
persists them, answers the idempotency question the ``PassRunner`` asks, and rolls
chunk-level signals up to one row per entry for the dashboard to query.

Two tables (both in the same LanceDB at ``config.DB_PATH``):

* ``signals``        — one row per ``(chunk_id, namespace, key)`` (the natural key).
* ``entry_signals``  — chunk->entry rollups, one row per ``(entry_id, namespace,
                       key)`` plus the entry's date, for cheap temporal queries.

LanceDB has no native upsert, so we delete-by-predicate then add — the same
delete-then-insert pattern the entries store uses for changed entries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pyarrow as pa

from . import config, store
from passes.deterministic import (
    Chunk,
    PassRunner,
    SignalRow,
    aggregate_emotion_to_entry,
    content_hash,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def signals_schema() -> pa.Schema:
    """Columns mirror ``SignalRow`` plus ``pass_name``/``pass_version`` bookkeeping.

    ``has()`` is keyed on pass name + version, and neither is recoverable from
    ``model_tag`` (which encodes the lexicon hash / model revision, not the pass
    version), so we stamp them as their own columns at upsert time.
    """
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("entry_id", pa.string()),
            pa.field("namespace", pa.string()),     # "emotion" | "lexical"
            pa.field("key", pa.string()),           # "goemotions" | "self_focus" | ...
            pa.field("value_num", pa.float64()),    # nullable
            pa.field("value_json", pa.string()),    # nullable, JSON-encoded dict
            pa.field("model_tag", pa.string()),
            pa.field("content_hash", pa.string()),  # chunk-text hash (passes' own)
            pa.field("computed_at", pa.string()),
            pa.field("pass_name", pa.string()),
            pa.field("pass_version", pa.string()),
        ]
    )


def entry_signals_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("date", pa.string()),          # ISO YYYY-MM-DD
            pa.field("date_int", pa.int64()),       # YYYYMMDD
            pa.field("namespace", pa.string()),
            pa.field("key", pa.string()),
            pa.field("value_num", pa.float64()),
            pa.field("value_json", pa.string()),
            pa.field("model_tag", pa.string()),
        ]
    )


def _open(db, name: str, schema: pa.Schema):
    db = db or store.connect()
    if name in store._table_names(db):
        return db.open_table(name)
    return db.create_table(name, schema=schema)


# --------------------------------------------------------------------------- #
# Signal store (satisfies the SignalStore protocol)
# --------------------------------------------------------------------------- #
class LanceSignalStore:
    """SignalStore protocol over LanceDB.

    Idempotency lives in ``PassRunner`` (it asks ``has()`` before computing). We
    keep a small in-memory set of already-computed
    ``(chunk_id, pass_name, pass_version, content_hash)`` keys so the per-chunk
    ``has()`` checks during a run are O(1) instead of re-scanning the table.
    """

    def __init__(self, db=None):
        self.db = db or store.connect()
        self.tbl = _open(self.db, config.SIGNALS_TABLE, signals_schema())
        self._seen: set[tuple[str, str, str, str]] | None = None
        self._current: tuple[str, str] | None = None   # (pass_name, pass_version)
        self._force = False                              # --rebuild: ignore has()

    # -- idempotency cache -------------------------------------------------- #
    def _load_seen(self) -> None:
        self._seen = set()
        try:
            df = store.table_to_df(self.tbl)
        except Exception:
            return
        if df.empty:
            return
        for cid, pn, pv, ch in zip(
            df["chunk_id"], df["pass_name"], df["pass_version"], df["content_hash"]
        ):
            self._seen.add((cid, pn, pv, ch))

    # -- protocol ----------------------------------------------------------- #
    def has(self, chunk_id: str, pass_name: str, pass_version: str,
            content_hash: str) -> bool:
        if self._force:
            return False
        if self._seen is None:
            self._load_seen()
        return (chunk_id, pass_name, pass_version, content_hash) in self._seen

    def upsert(self, rows: Sequence[SignalRow]) -> None:
        if not rows:
            return
        if self._current is None:
            # Fall back to deriving the pass name from model_tag; version unknown.
            pass_name = rows[0].model_tag.split(":", 1)[0]
            pass_version = ""
        else:
            pass_name, pass_version = self._current

        # Last-write-wins: clear this pass's existing rows for the touched chunks
        # (covers keys that may have disappeared between versions), then add.
        chunk_ids = sorted({r.chunk_id for r in rows})
        in_list = ", ".join(f"'{store.sql_quote(c)}'" for c in chunk_ids)
        self.tbl.delete(
            f"pass_name = '{store.sql_quote(pass_name)}' AND chunk_id IN ({in_list})"
        )

        records = [self._to_record(r, pass_name, pass_version) for r in rows]
        self.tbl.add(records)

        if self._seen is not None:
            for r in rows:
                self._seen.add((r.chunk_id, pass_name, pass_version, r.content_hash))

    @staticmethod
    def _to_record(r: SignalRow, pass_name: str, pass_version: str) -> dict:
        return {
            "chunk_id": r.chunk_id,
            "entry_id": r.entry_id,
            "namespace": r.namespace,
            "key": r.key,
            "value_num": None if r.value_num is None else float(r.value_num),
            "value_json": None if r.value_json is None else json.dumps(r.value_json),
            "model_tag": r.model_tag,
            "content_hash": r.content_hash,
            "computed_at": r.computed_at,
            "pass_name": pass_name,
            "pass_version": pass_version,
        }

    # -- runner glue -------------------------------------------------------- #
    def bind_pass(self, a_pass) -> None:
        """Record the active pass so ``upsert`` can stamp name/version."""
        self._current = (a_pass.name, a_pass.version)

    def run_pass(self, a_pass, chunks: Sequence[Chunk], *, force: bool = False,
                 batch_size: int = 64) -> int:
        """Bind the pass and run it through ``PassRunner`` against this store.

        ``force`` (CLI ``--rebuild``) makes ``has()`` return False so every chunk
        recomputes, without bumping any version.
        """
        self.bind_pass(a_pass)
        self._force = force
        try:
            return PassRunner(self).run(a_pass, chunks, batch_size=batch_size)
        finally:
            self._force = False

    def count(self) -> int:
        try:
            return self.tbl.count_rows()
        except Exception:
            return 0


# --------------------------------------------------------------------------- #
# Chunk loader: entries table -> Chunk(chunk_id, entry_id, text)
# --------------------------------------------------------------------------- #
def iter_chunks(tbl=None, date_from: str | None = None,
                date_to: str | None = None) -> list[Chunk]:
    """Load chunks from the entries table as ``Chunk`` objects, oldest first.

    The entries table stores one row per chunk; its ``id`` column is the chunk id
    (``"<entry_id>::<i>"``). Optional date prefilter mirrors ``stats.load_frame``.
    """
    tbl = tbl or store.open_or_create()
    try:
        df = store.table_to_df(tbl)
    except Exception:
        return []
    if df.empty:
        return []
    if date_from:
        df = df[df["date_int"] >= int(date_from.replace("-", ""))]
    if date_to:
        df = df[df["date_int"] <= int(date_to.replace("-", ""))]
    if df.empty:
        return []
    df = df.sort_values(["date_int", "entry_id", "chunk_index"])
    return [
        Chunk(chunk_id=str(r["id"]), entry_id=str(r["entry_id"]), text=str(r["text"]))
        for _, r in df.iterrows()
    ]


# --------------------------------------------------------------------------- #
# Phase 3: chunk -> entry aggregation (materialized into entry_signals)
# --------------------------------------------------------------------------- #
# Lexical rate keys roll up as token-weighted means; token_count sums; the
# temporal_focus dict rolls up per sub-key. Emotion uses the provided
# length-weighted mean + per-label peak helper.
_LEXICAL_RATE_KEYS = (
    "self_focus", "we_focus", "other_focus", "insight", "causation",
    "tentative", "certainty", "negation",
)


def _entry_meta(entries_df) -> dict[str, tuple[str, int]]:
    """entry_id -> (date, date_int) from the entries table."""
    out: dict[str, tuple[str, int]] = {}
    for eid, date, di in zip(entries_df["entry_id"], entries_df["date"],
                             entries_df["date_int"]):
        out.setdefault(str(eid), (str(date), int(di)))
    return out


def aggregate_entries(db=None, entries_tbl=None) -> int:
    """Rebuild the ``entry_signals`` rollup table from ``signals``.

    Fully derived from the signal store, so we rebuild it wholesale (delete-all
    then add) rather than tracking incremental deltas. Returns rows written.
    """
    db = db or store.connect()
    sig = LanceSignalStore(db)
    sdf = store.table_to_df(sig.tbl)
    entries_df = store.table_to_df(entries_tbl or store.open_or_create())
    out_tbl = _open(db, config.ENTRY_SIGNALS_TABLE, entry_signals_schema())

    if sdf.empty or entries_df.empty:
        try:
            out_tbl.delete("true")
        except Exception:
            pass
        return 0

    meta = _entry_meta(entries_df)
    word_count = {str(cid): float(wc or 1)
                  for cid, wc in zip(entries_df["id"], entries_df["word_count"])}

    records: list[dict] = []

    def emit(entry_id, namespace, key, *, value_num=None, value_json=None,
             model_tag=""):
        date, date_int = meta.get(entry_id, ("", 0))
        records.append({
            "entry_id": entry_id, "date": date, "date_int": date_int,
            "namespace": namespace, "key": key,
            "value_num": None if value_num is None else float(value_num),
            "value_json": None if value_json is None else json.dumps(value_json),
            "model_tag": model_tag,
        })

    # -- emotion: length-weighted mean + per-label peak --------------------- #
    emo = sdf[(sdf["namespace"] == "emotion") & (sdf["key"] == "goemotions")]
    for entry_id, grp in emo.groupby("entry_id"):
        vecs = [json.loads(v) for v in grp["value_json"] if v]
        if not vecs:
            continue
        weights = [word_count.get(str(cid), 1.0) for cid in grp["chunk_id"]]
        agg = aggregate_emotion_to_entry(vecs, weights)
        model_tag = grp["model_tag"].iloc[0]
        emit(str(entry_id), "emotion", "goemotions_mean",
             value_json=agg["mean"], model_tag=model_tag)
        emit(str(entry_id), "emotion", "goemotions_peak",
             value_json=agg["peak"], model_tag=model_tag)

    # -- lexical: token-weighted means ------------------------------------- #
    lex = sdf[sdf["namespace"] == "lexical"]
    if not lex.empty:
        # chunk_id -> token_count (the weight)
        tok = lex[lex["key"] == "token_count"]
        chunk_tokens = {str(c): float(v or 0.0)
                        for c, v in zip(tok["chunk_id"], tok["value_num"])}
        for entry_id, grp in lex.groupby("entry_id"):
            model_tag = grp["model_tag"].iloc[0]
            cids = list(grp["chunk_id"].unique())
            total_tok = sum(chunk_tokens.get(str(c), 0.0) for c in cids) or 1.0
            emit(str(entry_id), "lexical", "token_count",
                 value_num=total_tok, model_tag=model_tag)

            for key in _LEXICAL_RATE_KEYS:
                sub = grp[grp["key"] == key]
                if sub.empty:
                    continue
                num = sum(
                    float(v or 0.0) * chunk_tokens.get(str(c), 0.0)
                    for c, v in zip(sub["chunk_id"], sub["value_num"])
                )
                emit(str(entry_id), "lexical", key,
                     value_num=num / total_tok, model_tag=model_tag)

            temp = grp[grp["key"] == "temporal_focus"]
            if not temp.empty:
                acc = {"past": 0.0, "present": 0.0, "future": 0.0}
                for c, vj in zip(temp["chunk_id"], temp["value_json"]):
                    if not vj:
                        continue
                    w = chunk_tokens.get(str(c), 0.0)
                    d = json.loads(vj)
                    for k in acc:
                        acc[k] += float(d.get(k, 0.0)) * w
                rolled = {k: round(v / total_tok, 4) for k, v in acc.items()}
                emit(str(entry_id), "lexical", "temporal_focus",
                     value_json=rolled, model_tag=model_tag)

    try:
        out_tbl.delete("true")
    except Exception:
        pass
    if records:
        out_tbl.add(records)
    return len(records)


# --------------------------------------------------------------------------- #
# Dashboard query helpers (entry_signals -> tidy/wide pandas)
# --------------------------------------------------------------------------- #
def _entry_signals_df(db=None):
    db = db or store.connect()
    if config.ENTRY_SIGNALS_TABLE not in store._table_names(db):
        return None
    return store.table_to_df(db.open_table(config.ENTRY_SIGNALS_TABLE))


def signal_entry_count(db=None) -> int:
    """Distinct entries that have any rolled-up signal (for the dashboard gate)."""
    df = _entry_signals_df(db)
    if df is None or df.empty:
        return 0
    return int(df["entry_id"].nunique())


def emotion_entry_wide(db=None, which: str = "mean"):
    """One row per entry: entry_id, date, date_int + one column per emotion label.

    ``which`` selects the ``goemotions_mean`` or ``goemotions_peak`` rollup.
    """
    import pandas as pd

    df = _entry_signals_df(db)
    if df is None or df.empty:
        return pd.DataFrame()
    key = f"goemotions_{which}"
    sub = df[(df["namespace"] == "emotion") & (df["key"] == key)]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sub.iterrows():
        vec = json.loads(r["value_json"]) if r["value_json"] else {}
        rows.append({"entry_id": r["entry_id"], "date": r["date"],
                     "date_int": r["date_int"], **vec})
    return pd.DataFrame(rows).sort_values("date_int").reset_index(drop=True)


def lexical_entry_wide(db=None):
    """One row per entry: entry_id, date, date_int + each lexical rate column and
    past/present/future temporal fractions."""
    import pandas as pd

    df = _entry_signals_df(db)
    if df is None or df.empty:
        return pd.DataFrame()
    lex = df[df["namespace"] == "lexical"]
    if lex.empty:
        return pd.DataFrame()

    by_entry: dict[str, dict] = {}
    for _, r in lex.iterrows():
        eid = r["entry_id"]
        rec = by_entry.setdefault(
            eid, {"entry_id": eid, "date": r["date"], "date_int": r["date_int"]}
        )
        if r["key"] == "temporal_focus" and r["value_json"]:
            d = json.loads(r["value_json"])
            for k in ("past", "present", "future"):
                rec[f"temporal_{k}"] = d.get(k)
        elif r["value_num"] is not None:
            rec[r["key"]] = r["value_num"]
    return (
        pd.DataFrame(list(by_entry.values()))
        .sort_values("date_int")
        .reset_index(drop=True)
    )
