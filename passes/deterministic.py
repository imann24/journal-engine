"""
Deterministic derivation passes for the journal signal store.

Two passes, one contract:
  * GoEmotionsPass  -> 28-dim multi-label emotion vector per chunk (transformers, GPU)
  * LexicalPass     -> psycholinguistic rates per chunk (self-focus, insight, causation,
                       tentativeness, certainty, negation, temporal orientation)

Both are deterministic by construction: a classifier at fixed weights and lexicon
lookups have no sampling. Pin the model revision and the lexicon hash (both flow into
`model_tag`) and a re-run is byte-stable -- trends never shift under you because the
derivation changed silently.

Wire `SignalStore` to your LanceDB signals table; everything else is self-contained.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence, runtime_checkable


# --------------------------------------------------------------------------------------
# Signal-store contract
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    entry_id: str
    text: str


@dataclass(frozen=True)
class SignalRow:
    chunk_id: str
    entry_id: str
    namespace: str                  # "emotion" | "lexical"
    key: str                        # "goemotions" | "self_focus" | ...
    value_num: float | None = None
    value_json: dict | None = None
    model_tag: str = ""             # "goemotions:<rev>" | "lexical:v1+<lexhash>"
    content_hash: str = ""
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class Pass(Protocol):
    name: str
    version: str

    @property
    def model_tag(self) -> str: ...

    def compute(self, chunks: Sequence[Chunk]) -> list[SignalRow]: ...


@runtime_checkable
class SignalStore(Protocol):
    def has(self, chunk_id: str, pass_name: str, pass_version: str, content_hash: str) -> bool: ...
    def upsert(self, rows: Sequence[SignalRow]) -> None: ...


# --------------------------------------------------------------------------------------
# Runner -- idempotency lives here, not in the passes
# --------------------------------------------------------------------------------------

class PassRunner:
    """Run a pass over chunks, skipping anything already computed under the same
    (pass version, content hash). Mirrors the ingest dedup model: re-running only
    touches chunks whose text changed or whose pass version bumped."""

    def __init__(self, store: SignalStore):
        self.store = store

    def run(self, a_pass: Pass, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        pending = [
            c for c in chunks
            if not self.store.has(c.chunk_id, a_pass.name, a_pass.version, content_hash(c.text))
        ]
        written = 0
        for i in range(0, len(pending), batch_size):
            rows = a_pass.compute(pending[i:i + batch_size])
            self.store.upsert(rows)
            written += len(rows)
        return written


# --------------------------------------------------------------------------------------
# Pass 1 -- GoEmotions (28-dim multi-label emotion vector)
# --------------------------------------------------------------------------------------

GOEMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _windows(text: str, max_chars: int = 1800):
    """Greedily pack sentences into <=max_chars windows (~480 tokens at 512-limit).
    RoBERTa truncates at 512 tokens; a 400-word journal chunk often runs ~520, so
    naive truncation silently drops the tail. Window + average instead."""
    sents = _SENT_SPLIT.split(text.strip()) or [text]
    windows, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            windows.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        windows.append(cur)
    return windows or [text]


class GoEmotionsPass:
    name = "goemotions"
    version = "v1"

    def __init__(
        self,
        model_id: str = "SamLowe/roberta-base-go_emotions",
        revision: str = "main",      # pin to a commit hash for true reproducibility
        device: int | None = None,
        max_length: int = 512,
        fp16: bool = True,
    ):
        from transformers import pipeline
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._revision = revision
        self._max_length = max_length
        self.clf = pipeline(
            task="text-classification",
            model=model_id,
            revision=revision,
            top_k=None,              # return all 28 scores -> multi-label vector
            device=device,
            torch_dtype=torch.float16 if (fp16 and device != -1) else None,
        )

    @property
    def model_tag(self) -> str:
        return f"goemotions:{self._revision}"

    def _classify(self, texts: Sequence[str]) -> list[dict]:
        # Flatten every chunk into its windows, classify in one batched call,
        # then length-weighted-average the windows back per chunk.
        flat, spans = [], []
        for t in texts:
            ws = _windows(t)
            spans.append((len(flat), len(flat) + len(ws), [len(w) for w in ws]))
            flat.extend(ws)

        outputs = self.clf(
            flat, batch_size=len(flat) or 1,
            truncation=True, max_length=self._max_length,
        )
        scored = [{d["label"]: float(d["score"]) for d in row} for row in outputs]

        vecs = []
        for start, end, weights in spans:
            total = sum(weights) or 1
            acc = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
            for row, w in zip(scored[start:end], weights):
                for lbl, p in row.items():
                    acc[lbl] += p * w
            vecs.append({lbl: round(acc[lbl] / total, 5) for lbl in GOEMOTIONS_LABELS})
        return vecs

    def compute(self, chunks: Sequence[Chunk]) -> list[SignalRow]:
        vecs = self._classify([c.text for c in chunks])
        return [
            SignalRow(
                chunk_id=c.chunk_id, entry_id=c.entry_id,
                namespace="emotion", key="goemotions",
                value_json=vec, model_tag=self.model_tag,
                content_hash=content_hash(c.text),
            )
            for c, vec in zip(chunks, vecs)
        ]


# --------------------------------------------------------------------------------------
# Pass 2 -- Lexical / psycholinguistic rates
# --------------------------------------------------------------------------------------
# License-free word lists. These are the introspection-bearing categories: self-focus
# (first-person-singular rate) tracks self-attention/rumination; insight + causation
# track meaning-making; tentativeness vs certainty tracks cognitive stance. Edit freely
# -- the lexicons are hashed into model_tag, so any change is a new version and triggers
# a clean re-run of only this pass.

LEXICONS: dict[str, frozenset[str]] = {
    "first_person_singular": frozenset({"i", "me", "my", "mine", "myself"}),
    "first_person_plural": frozenset({"we", "us", "our", "ours", "ourselves"}),
    "other_person": frozenset({
        "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
        "they", "them", "their", "theirs",
    }),
    "insight": frozenset({
        "realize", "realise", "realized", "realised", "understand", "understood",
        "know", "knew", "think", "thought", "consider", "considered", "wonder",
        "wondered", "aware", "recognize", "recognise", "recognized", "sense",
        "figure", "figured", "reflect", "reflected", "notice", "noticed",
    }),
    "causation": frozenset({
        "because", "cause", "caused", "causes", "effect", "hence", "therefore",
        "thus", "since", "reason", "why", "consequently", "result", "results", "due",
    }),
    "tentative": frozenset({
        "maybe", "perhaps", "possibly", "probably", "might", "guess", "suppose",
        "seems", "seem", "seemed", "unsure", "somewhat", "fairly", "apparently",
    }),
    "certainty": frozenset({
        "always", "never", "definitely", "certainly", "sure", "clearly", "obviously",
        "absolutely", "completely", "totally", "undoubtedly", "must",
    }),
    "negation": frozenset({
        "no", "not", "never", "none", "nobody", "nothing", "neither", "nor",
        "cannot", "cant", "wont", "dont", "didnt", "isnt", "wasnt", "arent", "werent",
    }),
}

_FUTURE_MARKERS = frozenset({"will", "shall", "gonna", "going"})  # "going to" approx
_WORD = re.compile(r"[a-z']+")


def _hash_lexicons() -> str:
    blob = "|".join(f"{k}:{','.join(sorted(v))}" for k, v in sorted(LEXICONS.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


class LexicalPass:
    name = "lexical"
    version = "v1"

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                # POS + morph only; NER/lemmatizer/parser off for speed
                self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer", "parser"])
            except Exception:
                self.nlp = None  # graceful fallback: rates still work, tense is approximate
        self._lexhash = _hash_lexicons()

    @property
    def model_tag(self) -> str:
        return f"lexical:v1+{self._lexhash}"

    def compute(self, chunks: Sequence[Chunk]) -> list[SignalRow]:
        rows: list[SignalRow] = []
        for c in chunks:
            ch = content_hash(c.text)
            for key, val in self._features(c.text).items():
                rows.append(SignalRow(
                    chunk_id=c.chunk_id, entry_id=c.entry_id,
                    namespace="lexical", key=key,
                    value_num=val if not isinstance(val, dict) else None,
                    value_json=val if isinstance(val, dict) else None,
                    model_tag=self.model_tag, content_hash=ch,
                ))
        return rows

    def _features(self, text: str) -> dict:
        tokens = _WORD.findall(text.lower())
        n = len(tokens) or 1
        counts = {cat: 0 for cat in LEXICONS}
        for tok in tokens:
            for cat, lex in LEXICONS.items():
                if tok in lex:
                    counts[cat] += 1

        feats: dict = {
            "token_count": float(len(tokens)),
            "self_focus": counts["first_person_singular"] / n,
            "we_focus": counts["first_person_plural"] / n,
            "other_focus": counts["other_person"] / n,
            "insight": counts["insight"] / n,
            "causation": counts["causation"] / n,
            "tentative": counts["tentative"] / n,
            "certainty": counts["certainty"] / n,
            "negation": counts["negation"] / n,
            "temporal_focus": self._temporal_focus(text, tokens),
        }
        return {k: round(v, 5) if isinstance(v, float) else v for k, v in feats.items()}

    def _temporal_focus(self, text: str, tokens: list[str]) -> dict:
        """past / present / future fractions of tensed verbs.
        spaCy morph when available; otherwise a marker-word approximation."""
        if self.nlp is not None:
            past = pres = fut = 0
            doc = self.nlp(text)
            toks = list(doc)
            for i, t in enumerate(toks):
                if t.pos_ not in ("VERB", "AUX"):
                    continue
                low = t.text.lower()
                if low in _FUTURE_MARKERS:
                    fut += 1
                    continue
                tense = t.morph.get("Tense")
                if "Past" in tense:
                    past += 1
                elif "Pres" in tense:
                    pres += 1
            total = past + pres + fut or 1
            return {"past": round(past / total, 4),
                    "present": round(pres / total, 4),
                    "future": round(fut / total, 4)}

        # Fallback: crude markers only (no POS) -- flagged as approximate by lexhash.
        fut = sum(1 for t in tokens if t in _FUTURE_MARKERS)
        past = sum(1 for t in tokens if t.endswith("ed"))
        pres = max(len(tokens) // 20 - past - fut, 0)  # rough placeholder density
        total = past + pres + fut or 1
        return {"past": round(past / total, 4),
                "present": round(pres / total, 4),
                "future": round(fut / total, 4)}


# --------------------------------------------------------------------------------------
# Chunk -> entry rollup (length-weighted; keep mean and peak for emotion)
# --------------------------------------------------------------------------------------

def aggregate_emotion_to_entry(chunk_vecs: list[dict], weights: list[float]) -> dict:
    """Length-weighted mean (the entry's overall texture) plus per-label max
    (its emotional peak -- a single intense paragraph shouldn't be averaged away)."""
    total = sum(weights) or 1
    mean = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
    peak = {lbl: 0.0 for lbl in GOEMOTIONS_LABELS}
    for vec, w in zip(chunk_vecs, weights):
        for lbl in GOEMOTIONS_LABELS:
            mean[lbl] += vec.get(lbl, 0.0) * w
            peak[lbl] = max(peak[lbl], vec.get(lbl, 0.0))
    return {
        "mean": {lbl: round(mean[lbl] / total, 5) for lbl in GOEMOTIONS_LABELS},
        "peak": {lbl: round(peak[lbl], 5) for lbl in GOEMOTIONS_LABELS},
    }


# --------------------------------------------------------------------------------------
# Demo with an in-memory store
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    class MemStore:
        def __init__(self):
            self.rows: dict[tuple, SignalRow] = {}
            self.seen: set[tuple] = set()

        def has(self, chunk_id, pass_name, pass_version, content_hash):
            return (chunk_id, pass_name, pass_version, content_hash) in self.seen

        def upsert(self, rows):
            for r in rows:
                self.rows[(r.chunk_id, r.namespace, r.key)] = r
                self.seen.add((r.chunk_id, "goemotions" if r.namespace == "emotion" else "lexical",
                               "v1", r.content_hash))

    chunks = [
        Chunk("c1", "2021-03-04", "I keep replaying it. I think I understand now why I "
                                  "froze -- because I was terrified of being seen."),
        Chunk("c2", "2021-03-04", "We talked it through. Maybe things will be okay."),
    ]

    store = MemStore()
    runner = PassRunner(store)

    lex = LexicalPass()
    print("lexical rows:", runner.run(lex, chunks))
    for (cid, ns, key), row in store.rows.items():
        if ns == "lexical" and key in ("self_focus", "insight", "causation", "temporal_focus"):
            print(f"  {cid} {key:14} {row.value_num if row.value_num is not None else row.value_json}")

    # GoEmotions requires transformers + a model download; uncomment on the Spark:
    # emo = GoEmotionsPass(revision="main")
    # print("emotion rows:", runner.run(emo, chunks))
