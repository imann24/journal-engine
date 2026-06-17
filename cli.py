#!/usr/bin/env python3
"""Command-line interface for the journal engine.

    python cli.py ingest ./journals     # batch ingest a directory (idempotent)
    python cli.py ingest                 # sweep the configured drop folder
    python cli.py watch                  # auto-ingest drop folder on change
    python cli.py enrich [--limit N]     # LLM tagging pass (resumable)
    python cli.py search "money panic" --from 2019-01-01 --to 2019-12-31
    python cli.py ask "how did I talk about Max over time?"
    python cli.py stats

All paths/models/ports are configured via environment variables (see config.py
and .env.example).
"""

from __future__ import annotations

import argparse
import sys

from journal import config
from journal.enrich import enrich
from journal.ingest import ingest_dir
from journal.rag import ask
from journal.search import hybrid_search
from journal.stats import print_stats
from journal.watch import watch


def _print_ingest_summary(summary) -> None:
    print(f"\n{summary.n_entries} entries seen: "
          f"+{summary.added} added, ~{summary.updated} updated, "
          f"{summary.skipped} unchanged.")
    if summary.date_sources:
        print(f"Date sources (new/changed): {summary.date_sources}")
    frac = summary.mtime_fraction()
    if frac > config.MTIME_WARN_FRACTION:
        print(f"\n  WARNING: {frac:.0%} of new/changed entries fell back to file "
              "mtime for their date. Spot-check those before trusting temporal "
              "queries — consider adding a date to the filename or first line.")


def cmd_ingest(args) -> None:
    target = args.corpus_dir or config.DROP_DIR
    print(f"Ingesting from {target} ...")
    _print_ingest_summary(ingest_dir(target))


def cmd_enrich(args) -> None:
    def progress(n, total):
        if n % 10 == 0 or n == total:
            print(f"  {n}/{total} entries")
    n = enrich(limit=args.limit, progress=progress)
    print("Nothing to enrich." if n == 0 else f"Enriched {n} entries.")


def cmd_search(args) -> None:
    hits = hybrid_search(args.query, k=args.k,
                         date_from=args.date_from, date_to=args.date_to)
    if not hits:
        print("No matches.")
        return
    for h in hits:
        print(f"\n[{h['date']}] ({h['date_source']}) {h['entry_id']}")
        print(h["text"][:500])


def cmd_ask(args) -> None:
    ans = ask(args.question, k=args.k,
              date_from=args.date_from, date_to=args.date_to)
    print(ans.text)
    if ans.cited_dates:
        print(f"\n— drawn from entries dated: {', '.join(ans.cited_dates)}")


def cmd_stats(_args) -> None:
    print_stats()


def cmd_watch(args) -> None:
    watch(args.drop_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Local journal RAG + analytics engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="parse, date, chunk, embed, index (idempotent)")
    pi.add_argument("corpus_dir", nargs="?", default=None,
                    help="directory to ingest (defaults to the drop folder)")
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("enrich", help="LLM tagging pass for analytics (resumable)")
    pe.add_argument("--limit", type=int, default=None)
    pe.set_defaults(func=cmd_enrich)

    ps = sub.add_parser("search", help="hybrid retrieval, show raw hits")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=8)
    ps.add_argument("--from", dest="date_from", default=None)
    ps.add_argument("--to", dest="date_to", default=None)
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="RAG question answering with citations")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=8)
    pa.add_argument("--from", dest="date_from", default=None)
    pa.add_argument("--to", dest="date_to", default=None)
    pa.set_defaults(func=cmd_ask)

    pst = sub.add_parser("stats", help="temporal + thematic analytics")
    pst.set_defaults(func=cmd_stats)

    pw = sub.add_parser("watch", help="auto-ingest the drop folder on change")
    pw.add_argument("drop_dir", nargs="?", default=None)
    pw.set_defaults(func=cmd_watch)

    args = p.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
