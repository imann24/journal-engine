#!/usr/bin/env python3
"""
journal_engine.py — a fully local history / analysis / ingestion / query engine
for a corpus of .txt journal entries.

Stack:
  - Embeddings & generation: Ollama (BGE-M3 + your Nemotron/Qwen model)
  - Vector store + full-text search: LanceDB (embedded, no server)
  - Retrieval: hybrid dense+sparse with reciprocal-rank fusion + date prefilter

Designed to run on the DGX Spark. Query from the Mac Studio over Tailscale by
exporting OLLAMA_HOST=http://<spark-tailscale-name>:11434 before running.

Setup:
    pip install lancedb ollama python-dateutil pandas tqdm
    ollama pull bge-m3
    ollama pull nemotron        # or whatever chat model you run

Usage:
    python journal_engine.py ingest  ./journals
    python journal_engine.py enrich                      # one-time LLM tagging pass
    python journal_engine.py search "panic about money" --from 2019-01-01 --to 2019-12-31
    python journal_engine.py ask     "How did I talk about Max over time?"
    python journal_engine.py stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import lancedb
import ollama
import pandas as pd
from dateutil import parser as dateparser

# --------------------------------------------------------------------------- #
# Config — change model tags to match what you have pulled in Ollama.
# --------------------------------------------------------------------------- #
DB_PATH = os.environ.get("JOURNAL_DB", "./journal_lancedb")
TABLE = "entries"
EMBED_MODEL = os.environ.get("JOURNAL_EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.environ.get("JOURNAL_CHAT_MODEL", "nemotron")
CHUNK_TARGET_WORDS = 400      # split entries longer than this
CHUNK_OVERLAP_PARAS = 1       # paragraphs of overlap between sub-chunks
EMBED_BATCH = 64

# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings via Ollama."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = ollama.embed(model=EMBED_MODEL, input=batch)
        out.extend(resp["embeddings"])
    return out


def chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Date inference — the part that quietly makes or breaks the whole thing.
# Tries: filename pattern -> header line in body -> file modification time.
# --------------------------------------------------------------------------- #
_FILENAME_DATE_RES = [
    re.compile(r"(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})"),   # 2013-05-04, 2013_5_4
    re.compile(r"(\d{4})(\d{2})(\d{2})"),                    # 20130504
]
_TEXTUAL_DATE_RE = re.compile(
    r"\b("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"                          # 2013-05-04
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)


def infer_date(path: Path, body: str) -> tuple[date, str]:
    """Return (date, source_of_date) for an entry."""
    name = path.stem
    for rx in _FILENAME_DATE_RES:
        m = rx.search(name)
        if m:
            try:
                y, mo, d = (int(g) for g in m.groups())
                if 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return date(y, mo, d), "filename"
            except ValueError:
                pass

    head = "\n".join(body.splitlines()[:4])
    m = _TEXTUAL_DATE_RE.search(head)
    if m:
        try:
            return dateparser.parse(m.group(1), fuzzy=True).date(), "header"
        except (ValueError, OverflowError):
            pass

    return datetime.fromtimestamp(path.stat().st_mtime).date(), "mtime"


# --------------------------------------------------------------------------- #
# Ingestion + chunking
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: str
    entry_id: str
    source: str
    date: str          # ISO YYYY-MM-DD
    date_int: int      # YYYYMMDD, for cheap range filters
    date_source: str
    text: str
    word_count: int
    vector: list[float] = field(default_factory=list)
    # enrichment columns (filled later)
    mood: int = 0
    topics: str = ""
    people: str = ""
    places: str = ""


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _chunk_body(body: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        return []
    if sum(len(p.split()) for p in paras) <= CHUNK_TARGET_WORDS:
        return ["\n\n".join(paras)]

    chunks, cur, cur_words = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur_words + w > CHUNK_TARGET_WORDS and cur:
            chunks.append("\n\n".join(cur))
            cur = cur[-CHUNK_OVERLAP_PARAS:] if CHUNK_OVERLAP_PARAS else []
            cur_words = sum(len(c.split()) for c in cur)
        cur.append(p)
        cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def ingest(corpus_dir: str) -> None:
    root = Path(corpus_dir).expanduser()
    files = sorted(root.rglob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found under {root}")

    print(f"Found {len(files)} files. Parsing + dating...")
    chunks: list[Chunk] = []
    date_sources = {"filename": 0, "header": 0, "mtime": 0}

    for f in files:
        body = _read_text(f).strip()
        if not body:
            continue
        d, dsrc = infer_date(f, body)
        date_sources[dsrc] += 1
        entry_id = str(f.relative_to(root))
        for i, ctext in enumerate(_chunk_body(body)):
            chunks.append(
                Chunk(
                    id=f"{entry_id}::{i}",
                    entry_id=entry_id,
                    source=str(f),
                    date=d.isoformat(),
                    date_int=int(d.strftime("%Y%m%d")),
                    date_source=dsrc,
                    text=ctext,
                    word_count=len(ctext.split()),
                )
            )

    print(f"  {len(chunks)} chunks. Date sources: {date_sources}")
    if date_sources["mtime"] > len(files) * 0.3:
        print("  WARNING: many entries fell back to file mtime. Spot-check those "
              "dates before trusting temporal queries.")

    print(f"Embedding with {EMBED_MODEL}...")
    vectors = embed_texts([c.text for c in chunks])
    for c, v in zip(chunks, vectors):
        c.vector = v

    db = lancedb.connect(DB_PATH)
    rows = [c.__dict__ for c in chunks]
    if TABLE in db.table_names():
        db.drop_table(TABLE)
    tbl = db.create_table(TABLE, data=rows)
    tbl.create_fts_index("text", replace=True)
    print(f"Indexed {len(rows)} chunks into {DB_PATH}/{TABLE}. Ready to query.")


# --------------------------------------------------------------------------- #
# Retrieval — hybrid (vector + BM25) fused with reciprocal-rank fusion.
# --------------------------------------------------------------------------- #
def _date_filter(date_from: str | None, date_to: str | None) -> str | None:
    clauses = []
    if date_from:
        clauses.append(f"date_int >= {int(date_from.replace('-', ''))}")
    if date_to:
        clauses.append(f"date_int <= {int(date_to.replace('-', ''))}")
    return " AND ".join(clauses) if clauses else None


def hybrid_search(
    query: str,
    k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    pool: int = 40,
    rrf_k: int = 60,
) -> list[dict]:
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE)
    flt = _date_filter(date_from, date_to)

    qvec = embed_texts([query])[0]
    vq = tbl.search(qvec).limit(pool)
    if flt:
        vq = vq.where(flt, prefilter=True)
    dense = vq.to_list()

    fq = tbl.search(query, query_type="fts").limit(pool)
    if flt:
        fq = fq.where(flt, prefilter=True)
    sparse = fq.to_list()

    scores: dict[str, float] = {}
    store: dict[str, dict] = {}
    for ranked in (dense, sparse):
        for rank, row in enumerate(ranked):
            cid = row["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            store[cid] = row
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [store[cid] for cid in top]


# --------------------------------------------------------------------------- #
# RAG question answering
# --------------------------------------------------------------------------- #
RAG_SYSTEM = (
    "You are an assistant that answers questions strictly from the user's own "
    "journal excerpts provided below. Quote or paraphrase only what is present. "
    "Always attribute claims to their entry date, e.g. (2019-07-14). If the "
    "excerpts don't cover the question, say so plainly."
)


def ask(question: str, k: int = 8, date_from=None, date_to=None) -> str:
    hits = hybrid_search(question, k=k, date_from=date_from, date_to=date_to)
    if not hits:
        return "No relevant entries found for that query/date range."
    context = "\n\n".join(
        f"[{h['date']}] {h['text']}" for h in sorted(hits, key=lambda r: r["date"])
    )
    prompt = f"JOURNAL EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
    answer = chat(prompt, system=RAG_SYSTEM)
    cited = ", ".join(sorted({h["date"] for h in hits}))
    return f"{answer}\n\n— drawn from entries dated: {cited}"


# --------------------------------------------------------------------------- #
# Enrichment — one-time LLM pass that tags each entry for analytics.
# Idempotent-ish: only re-tags chunks whose mood is still 0.
# --------------------------------------------------------------------------- #
ENRICH_SYSTEM = (
    "Extract structured metadata from a single journal excerpt. Respond with "
    "ONLY a JSON object, no prose, with keys: "
    "mood (integer 1-5, 1=very low, 5=very high), "
    "topics (array of 1-4 short lowercase theme tags), "
    "people (array of first names mentioned), "
    "places (array of place names mentioned)."
)


def _safe_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def enrich(limit: int | None = None) -> None:
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE)
    df = tbl.to_pandas()
    todo = df[df["mood"] == 0]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print("Nothing to enrich.")
        return

    print(f"Enriching {len(todo)} chunks with {CHAT_MODEL}...")
    updates = []
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        meta = _safe_json(chat(row["text"][:4000], system=ENRICH_SYSTEM))
        updates.append(
            {
                "id": row["id"],
                "mood": int(meta.get("mood", 3) or 3),
                "topics": ", ".join(meta.get("topics", []) or []),
                "people": ", ".join(meta.get("people", []) or []),
                "places": ", ".join(meta.get("places", []) or []),
            }
        )
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}")

    for u in updates:
        tbl.update(
            where=f"id = '{u['id'].replace(chr(39), chr(39) * 2)}'",
            values={k: v for k, v in u.items() if k != "id"},
        )
    print("Enrichment complete.")


# --------------------------------------------------------------------------- #
# Analytics — pure dataframe queries over the enriched columns, no LLM needed.
# --------------------------------------------------------------------------- #
def stats() -> None:
    db = lancedb.connect(DB_PATH)
    df = db.open_table(TABLE).to_pandas()
    df["year"] = df["date"].str[:4]

    print("\n=== Entries (chunks) per year ===")
    print(df.groupby("year").size().to_string())

    if (df["mood"] > 0).any():
        print("\n=== Mean mood per year (1-5) ===")
        m = df[df["mood"] > 0].groupby("year")["mood"].mean().round(2)
        print(m.to_string())

        print("\n=== Most-mentioned people ===")
        print(_top_tokens(df["people"]).to_string())

        print("\n=== Most-mentioned places ===")
        print(_top_tokens(df["places"]).to_string())

        print("\n=== Top topics ===")
        print(_top_tokens(df["topics"]).to_string())
    else:
        print("\n(Run `enrich` to unlock mood/people/place/topic analytics.)")


def _top_tokens(series: pd.Series, n: int = 12) -> pd.Series:
    tokens = (
        series.dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    return tokens.value_counts().head(n)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Local journal RAG + analytics engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="parse, date, chunk, embed, index")
    pi.add_argument("corpus_dir")

    pe = sub.add_parser("enrich", help="LLM tagging pass for analytics")
    pe.add_argument("--limit", type=int, default=None)

    ps = sub.add_parser("search", help="hybrid retrieval, show raw hits")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=8)
    ps.add_argument("--from", dest="date_from", default=None)
    ps.add_argument("--to", dest="date_to", default=None)

    pa = sub.add_parser("ask", help="RAG question answering")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=8)
    pa.add_argument("--from", dest="date_from", default=None)
    pa.add_argument("--to", dest="date_to", default=None)

    sub.add_parser("stats", help="temporal + thematic analytics")

    args = p.parse_args()

    if args.cmd == "ingest":
        ingest(args.corpus_dir)
    elif args.cmd == "enrich":
        enrich(limit=args.limit)
    elif args.cmd == "search":
        for h in hybrid_search(args.query, k=args.k,
                               date_from=args.date_from, date_to=args.date_to):
            print(f"\n[{h['date']}] ({h['date_source']}) {h['entry_id']}")
            print(h["text"][:500])
    elif args.cmd == "ask":
        print(ask(args.question, k=args.k,
                  date_from=args.date_from, date_to=args.date_to))
    elif args.cmd == "stats":
        stats()


if __name__ == "__main__":
    main()
