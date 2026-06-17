"""Drop-folder watcher: ingest .txt files as they are created/modified in the
configured drop directory. `ingest <dir>` is the equivalent manual sweep.
"""

from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from . import config
from .ingest import EntryRecord, ingest_records, read_text


def _ingest_path(path: Path, root: Path) -> None:
    try:
        body = read_text(path).strip()
    except OSError:
        return
    if not body:
        return
    try:
        entry_id = str(path.relative_to(root))
    except ValueError:
        entry_id = str(path)
    summary = ingest_records(
        [EntryRecord(entry_id=entry_id, body=body, source=str(path), path=path)]
    )
    for r in summary.results:
        if r.status in ("added", "updated"):
            print(f"  [{r.status}] {r.entry_id}  ({r.date}, {r.date_source}, "
                  f"{r.n_chunks} chunk(s))")
        elif r.status == "skipped":
            print(f"  [skipped, unchanged] {r.entry_id}")


class _Handler(FileSystemEventHandler):
    def __init__(self, root: Path):
        self.root = root
        # Debounce rapid duplicate events per path.
        self._last: dict[str, float] = {}

    def _maybe(self, path_str: str) -> None:
        if not path_str.endswith(".txt"):
            return
        now = time.time()
        if now - self._last.get(path_str, 0) < 1.0:
            return
        self._last[path_str] = now
        _ingest_path(Path(path_str), self.root)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)


def watch(drop_dir: str | None = None) -> None:
    root = Path(drop_dir or config.DROP_DIR).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Watching {root} for new/modified .txt files. Ctrl-C to stop.")
    print("Doing an initial sweep...")
    from .ingest import ingest_dir
    summary = ingest_dir(str(root))
    print(f"  swept: +{summary.added} added, ~{summary.updated} updated, "
          f"{summary.skipped} unchanged.")

    observer = Observer()
    observer.schedule(_Handler(root), str(root), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher.")
        observer.stop()
    observer.join()
