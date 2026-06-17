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

from journal import config, store
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
    n = enrich(limit=args.limit, progress=progress, model=args.model)
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
              date_from=args.date_from, date_to=args.date_to, model=args.model)
    print(ans.text)
    if ans.cited_dates:
        print(f"\n— drawn from entries dated: {', '.join(ans.cited_dates)}")


def cmd_stats(_args) -> None:
    print_stats()


def cmd_list(_args) -> None:
    df = store.list_entries(store.open_or_create())
    if df.empty:
        print("No entries indexed.")
        return
    print(f"{len(df)} entries:\n")
    for _, r in df.iterrows():
        print(f"  {r['date']}  [{r['date_source']:>8}]  {r['chunks']:>2} chunk(s)  "
              f"{r['entry_id']}")


def cmd_remove(args) -> None:
    tbl = store.open_or_create()

    if args.all:
        if not args.yes and input("Delete ALL entries? type 'yes' to confirm: ").strip() != "yes":
            print("Aborted.")
            return
        store.delete_all(tbl)
        print("Deleted all entries.")
        return

    ids = list(args.entry_ids)
    if args.date_from or args.date_to:
        ids += store.entry_ids_in_range(tbl, args.date_from, args.date_to)
    ids = sorted(set(i for i in ids if i))

    if not ids:
        print("Nothing to remove. Pass entry id(s), --from/--to, or --all. "
              "Use `list` to see entry ids.")
        return

    print(f"About to remove {len(ids)} entr{'y' if len(ids) == 1 else 'ies'}:")
    for i in ids[:20]:
        print(f"  - {i}")
    if len(ids) > 20:
        print(f"  ... and {len(ids) - 20} more")
    if not args.yes and input("Confirm? type 'yes': ").strip() != "yes":
        print("Aborted.")
        return
    n = store.delete_entries(tbl, ids)
    print(f"Removed {n} entries.")


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
    pe.add_argument("--model", default=None, help="Ollama model tag (overrides default)")
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
    pa.add_argument("--model", default=None, help="Ollama model tag (overrides default)")
    pa.set_defaults(func=cmd_ask)

    pst = sub.add_parser("stats", help="temporal + thematic analytics")
    pst.set_defaults(func=cmd_stats)

    pl = sub.add_parser("list", help="list indexed entries and their ids")
    pl.set_defaults(func=cmd_list)

    prm = sub.add_parser("remove", help="remove entries by id, date range, or all")
    prm.add_argument("entry_ids", nargs="*", help="entry id(s) to remove (see `list`)")
    prm.add_argument("--from", dest="date_from", default=None,
                     help="remove entries on/after this date (YYYY-MM-DD)")
    prm.add_argument("--to", dest="date_to", default=None,
                     help="remove entries on/before this date (YYYY-MM-DD)")
    prm.add_argument("--all", action="store_true", help="remove ALL entries")
    prm.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    prm.set_defaults(func=cmd_remove)

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
