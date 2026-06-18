"""Signal store tests: runner idempotency, lexical feature rates, and the
chunk->entry rollup.

No Ollama, no GPU, no transformers are required: the lexical pass runs with
spaCy disabled (rates are POS-independent), emotion signals are synthesized as
plain SignalRows, and everything else is real LanceDB I/O on a tmp DB.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from journal import config, ingest as ingest_mod, signal_store as ss, store
from passes.deterministic import (
    Chunk,
    GOEMOTIONS_LABELS,
    LexicalPass,
    SignalRow,
    aggregate_emotion_to_entry,
    content_hash,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "db"))
    monkeypatch.setattr(config, "EMBED_DIM", 8)

    def fake_embed(texts):
        return [[float(len(t) % 7)] * 8 for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)
    return tmp_path


class FakePass:
    """Minimal Pass: one numeric signal per chunk; records which chunks it saw."""

    def __init__(self, version: str = "v1", value: float = 1.0):
        self.name = "fake"
        self.version = version
        self._value = value
        self.seen: list[str] = []

    @property
    def model_tag(self) -> str:
        return f"fake:{self.version}"

    def compute(self, chunks):
        self.seen.extend(c.chunk_id for c in chunks)
        return [
            SignalRow(
                chunk_id=c.chunk_id, entry_id=c.entry_id, namespace="fake",
                key="v", value_num=self._value, model_tag=self.model_tag,
                content_hash=content_hash(c.text),
            )
            for c in chunks
        ]


# --------------------------------------------------------------------------- #
# 1. Runner dedup / idempotency
# --------------------------------------------------------------------------- #
def test_runner_idempotency(fresh_db):
    chunks = [Chunk("c1", "e1", "alpha text"), Chunk("c2", "e1", "beta text")]

    # First run writes both; second run (fresh store handle, reads disk) writes 0.
    assert ss.LanceSignalStore().run_pass(FakePass(), chunks) == 2
    p2 = FakePass()
    assert ss.LanceSignalStore().run_pass(p2, chunks) == 0
    assert p2.seen == []  # nothing recomputed

    # Editing one chunk's text re-derives only that chunk.
    edited = [Chunk("c1", "e1", "alpha text CHANGED"), Chunk("c2", "e1", "beta text")]
    p3 = FakePass()
    assert ss.LanceSignalStore().run_pass(p3, edited) == 1
    assert p3.seen == ["c1"]

    # Bumping the pass version re-derives everything.
    p4 = FakePass(version="v2")
    assert ss.LanceSignalStore().run_pass(p4, edited) == 2
    assert sorted(p4.seen) == ["c1", "c2"]


def test_force_rebuild_recomputes(fresh_db):
    chunks = [Chunk("c1", "e1", "alpha"), Chunk("c2", "e1", "beta")]
    ss.LanceSignalStore().run_pass(FakePass(), chunks)
    p = FakePass()
    assert ss.LanceSignalStore().run_pass(p, chunks, force=True) == 2
    assert sorted(p.seen) == ["c1", "c2"]


def test_upsert_last_write_wins(fresh_db):
    chunks = [Chunk("c1", "e1", "alpha")]
    ss.LanceSignalStore().run_pass(FakePass(value=1.0), chunks, force=True)
    ss.LanceSignalStore().run_pass(FakePass(value=9.0), chunks, force=True)
    df = store.table_to_df(ss.LanceSignalStore().tbl)
    fake = df[df["namespace"] == "fake"]
    assert len(fake) == 1                       # no duplicate on the natural key
    assert fake.iloc[0]["value_num"] == 9.0     # newest value wins


# --------------------------------------------------------------------------- #
# 2. Lexical feature rates (spaCy disabled -> POS-independent)
# --------------------------------------------------------------------------- #
def test_lexical_feature_rates():
    feats = LexicalPass(use_spacy=False)._features("I think I understand because I tried.")
    # tokens: i think i understand because i tried  (n = 7)
    assert feats["token_count"] == 7.0
    assert feats["self_focus"] == pytest.approx(3 / 7, abs=1e-5)     # i, i, i
    assert feats["insight"] == pytest.approx(2 / 7, abs=1e-5)        # think, understand
    assert feats["causation"] == pytest.approx(1 / 7, abs=1e-5)      # because
    assert feats["certainty"] == 0.0
    # temporal_focus is a dict that always sums to ~1 over the three tenses.
    t = feats["temporal_focus"]
    assert set(t) == {"past", "present", "future"}
    assert sum(t.values()) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 3. Chunk -> entry rollup
# --------------------------------------------------------------------------- #
def test_entry_rollup(fresh_db):
    # A two-paragraph entry chunks into multiple chunks (each paragraph > target).
    para = " ".join(["alpha"] * 300)
    para2 = " ".join(["beta"] * 300)
    ingest_mod.ingest_paste(f"{para}\n\n{para2}", "2020-01-01")

    tbl = store.open_or_create()
    edf = store.table_to_df(tbl)
    assert len(edf) >= 2, "entry should produce multiple chunks for the rollup"
    entry_id = edf["entry_id"].iloc[0]

    # --- emotion: synthesize a distinct vector per chunk, weight by word_count --
    sstore = ss.LanceSignalStore()
    sstore.bind_pass(SimpleNamespace(name="goemotions", version="v1"))
    vecs, weights, rows = [], [], []
    for i, (cid, wc) in enumerate(zip(edf["id"], edf["word_count"]), start=1):
        vec = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
        vec["joy"] = round(0.1 * i, 5)
        vec["anger"] = round(0.05 * i, 5)
        vecs.append(vec)
        weights.append(float(wc))
        rows.append(SignalRow(
            chunk_id=str(cid), entry_id=str(entry_id), namespace="emotion",
            key="goemotions", value_json=vec,
            model_tag=f"goemotions:{config.GOEMOTIONS_REVISION}",
            content_hash=content_hash(str(cid)),
        ))
    sstore.upsert(rows)

    # --- lexical: real pass over the actual chunks ----------------------------
    ss.LanceSignalStore().run_pass(LexicalPass(use_spacy=False), ss.iter_chunks())

    assert ss.aggregate_entries() > 0

    # Emotion rollup == provided length-weighted helper.
    expected = aggregate_emotion_to_entry(vecs, weights)
    emo = ss.emotion_entry_wide(which="mean")
    assert len(emo) == 1
    assert emo.iloc[0]["joy"] == pytest.approx(expected["mean"]["joy"], abs=1e-4)
    assert emo.iloc[0]["anger"] == pytest.approx(expected["mean"]["anger"], abs=1e-4)

    # Lexical rollup == token-weighted mean of per-chunk rates.
    lp = LexicalPass(use_spacy=False)
    feats = [lp._features(str(t)) for t in edf.sort_values("chunk_index")["text"]]
    toks = [f["token_count"] for f in feats]
    exp_self = sum(f["self_focus"] * t for f, t in zip(feats, toks)) / sum(toks)
    lex = ss.lexical_entry_wide()
    assert len(lex) == 1
    assert lex.iloc[0]["self_focus"] == pytest.approx(exp_self, abs=1e-4)
    assert lex.iloc[0]["token_count"] == pytest.approx(sum(toks), abs=1e-3)
