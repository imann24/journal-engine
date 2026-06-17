"""Ingestion pipeline — one code path for every source (files, pasted text,
batch paste, uploaded .txt). Idempotent and incremental:

    new entry      -> chunk, embed, insert            (status "added")
    changed entry  -> delete old chunks, re-insert    (status "updated")
    unchanged entry-> skip entirely                    (status "skipped")

Only added/updated entries are embedded, and they are re-flagged unenriched so
`enrich` re-tags exactly what changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from . import config, store
from .chunking import chunk_body
from .dating import infer_date_for_file, infer_date_for_text, to_date_int
from .embeddings import embed_texts


@dataclass
class EntryRecord:
    """A single raw entry handed to the pipeline, regardless of where it came
    from. Exactly one of (path) or (explicit text source) is the origin."""

    entry_id: str
    body: str
    source: str
    path: Path | None = None
    explicit_date: str | date | None = None


@dataclass
class IngestResult:
    entry_id: str
    status: str            # added | updated | skipped | empty
    date: str = ""
    date_source: str = ""
    n_chunks: int = 0


@dataclass
class IngestSummary:
    results: list[IngestResult] = field(default_factory=list)
    date_sources: dict[str, int] = field(default_factory=dict)

    @property
    def added(self) -> int:
        return sum(r.status == "added" for r in self.results)

    @property
    def updated(self) -> int:
        return sum(r.status == "updated" for r in self.results)

    @property
    def skipped(self) -> int:
        return sum(r.status == "skipped" for r in self.results)

    @property
    def n_entries(self) -> int:
        return sum(r.status in ("added", "updated", "skipped") for r in self.results)

    def mtime_fraction(self) -> float:
        dated = [r for r in self.results if r.status in ("added", "updated")]
        if not dated:
            return 0.0
        return sum(r.date_source == "mtime" for r in dated) / len(dated)


# --------------------------------------------------------------------------- #
# File reading
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Building EntryRecords from the various sources
# --------------------------------------------------------------------------- #
def records_from_dir(corpus_dir: str) -> list[EntryRecord]:
    root = Path(corpus_dir).expanduser()
    records: list[EntryRecord] = []
    for f in sorted(root.rglob("*.txt")):
        body = read_text(f).strip()
        if not body:
            continue
        # entry_id is the path relative to root when possible, else absolute —
        # stable across re-runs so dedup works.
        try:
            entry_id = str(f.relative_to(root))
        except ValueError:
            entry_id = str(f)
        records.append(EntryRecord(entry_id=entry_id, body=body, source=str(f), path=f))
    return records


def record_from_paste(body: str, explicit_date: str | None = None,
                      source: str = "paste") -> EntryRecord:
    """A single pasted/typed entry. entry_id is content-derived so identical
    pastes dedup, but an explicit date is honoured for the dating step."""
    h = store.content_hash(body)[:16]
    return EntryRecord(entry_id=f"{source}:{h}", body=body, source=source,
                       explicit_date=explicit_date)


# Batch paste: split on a line that is just "---" or "===", or on 2+ blank lines.
_BATCH_SPLIT_RE = re.compile(r"\n\s*(?:-{3,}|={3,})\s*\n|\n{3,}")


def records_from_batch(blob: str, source: str = "batch") -> list[EntryRecord]:
    parts = [p.strip() for p in _BATCH_SPLIT_RE.split(blob) if p.strip()]
    return [record_from_paste(p, source=source) for p in parts]


def record_from_upload(filename: str, body: str) -> EntryRecord:
    """An uploaded .txt: use the filename for both id and date inference."""
    body = body.strip()
    return EntryRecord(entry_id=f"upload:{filename}", body=body,
                       source=f"upload:{filename}", path=Path(filename))


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _resolve_date(rec: EntryRecord) -> tuple[date, str]:
    # A real file on disk -> filename/header/mtime. Uploads have a filename but
    # no real path/mtime, so they use the text path (filename header / today).
    if rec.path is not None and rec.path.exists():
        return infer_date_for_file(rec.path, rec.body)
    if rec.path is not None:
        # Uploaded file: try its filename, then header, then today.
        from .dating import date_from_filename
        got = date_from_filename(rec.path.stem)
        if got:
            return got, "filename"
        return infer_date_for_text(rec.body)
    return infer_date_for_text(rec.body, explicit=rec.explicit_date)


def ingest_records(records: list[EntryRecord], tbl=None,
                  rebuild_index: bool = True) -> IngestSummary:
    tbl = tbl or store.open_or_create()
    existing = store.existing_entry_hashes(tbl)
    summary = IngestSummary()
    now = datetime.now().isoformat(timespec="seconds")

    pending_rows: list[dict] = []   # rows awaiting embedding+insert
    to_delete: list[str] = []       # entry_ids whose old chunks must go

    for rec in records:
        body = rec.body.strip()
        if not body:
            summary.results.append(IngestResult(rec.entry_id, "empty"))
            continue

        chash = store.content_hash(body)
        prior = existing.get(rec.entry_id)
        if prior == chash:
            summary.results.append(IngestResult(rec.entry_id, "skipped"))
            continue

        d, dsrc = _resolve_date(rec)
        summary.date_sources[dsrc] = summary.date_sources.get(dsrc, 0) + 1
        status = "updated" if prior is not None else "added"
        if prior is not None:
            to_delete.append(rec.entry_id)

        chunks = chunk_body(body)
        for i, ctext in enumerate(chunks):
            pending_rows.append(
                {
                    "id": f"{rec.entry_id}::{i}",
                    "entry_id": rec.entry_id,
                    "source": rec.source,
                    "date": d.isoformat(),
                    "date_int": to_date_int(d),
                    "date_source": dsrc,
                    "text": ctext,
                    "word_count": len(ctext.split()),
                    "content_hash": chash,
                    "chunk_index": i,
                    "ingested_at": now,
                    "mood": 0,
                    "topics": "",
                    "people": "",
                    "places": "",
                    "enriched": False,
                }
            )
        summary.results.append(
            IngestResult(rec.entry_id, status, d.isoformat(), dsrc, len(chunks))
        )

    # Delete superseded chunks for changed entries.
    for entry_id in to_delete:
        store.delete_entry(tbl, entry_id)

    # Embed all new/changed chunks in one batched pass, then insert.
    if pending_rows:
        vectors = embed_texts([r["text"] for r in pending_rows])
        for r, v in zip(pending_rows, vectors):
            r["vector"] = v
        tbl.add(pending_rows)
        if rebuild_index:
            store.rebuild_fts(tbl)

    return summary


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the CLI and web UI
# --------------------------------------------------------------------------- #
def ingest_dir(corpus_dir: str, tbl=None) -> IngestSummary:
    return ingest_records(records_from_dir(corpus_dir), tbl=tbl)


def ingest_paste(body: str, explicit_date: str | None = None, tbl=None) -> IngestSummary:
    return ingest_records([record_from_paste(body, explicit_date)], tbl=tbl)


def ingest_batch(blob: str, tbl=None) -> IngestSummary:
    return ingest_records(records_from_batch(blob), tbl=tbl)


def ingest_uploads(files: list[tuple[str, str]], tbl=None) -> IngestSummary:
    """files: list of (filename, text_content)."""
    recs = [record_from_upload(name, content) for name, content in files]
    return ingest_records(recs, tbl=tbl)
