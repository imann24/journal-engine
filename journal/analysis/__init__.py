"""Analysis layer over the signal store — INTERFACES + TODOs ONLY, not built.

The signal store (``journal.signal_store``) turns the corpus into queryable typed
signals. This package is where *analysis* modules live: higher-order reads that
find structure across time, on top of the signal store and the existing
``bge-m3`` vectors. Nothing here is implemented yet, and **no new dependencies
have been added** — installing ``ruptures`` / ``umap-learn`` / ``hdbscan``
requires approval (see CLAUDE.md's approved-dependency list).

Two planned modules:

1. Change-point detection (``ruptures``)
   --------------------------------------
   Given a per-period signal series from ``entry_signals`` (e.g. ``self_focus``
   or a single emotion label over months), detect dates where the level shifts —
   "something changed here" markers for the dashboard timelines.

   def detect_change_points(series: "pd.Series", penalty: float = 3.0) -> list:
       '''Return period indices where the signal's mean shifts (ruptures PELT).'''
       raise NotImplementedError

2. Theme discovery (UMAP + HDBSCAN over bge-m3 vectors)
   ----------------------------------------------------
   The entries table already stores 1024-dim ``bge-m3`` vectors per chunk. Reduce
   with UMAP, cluster with HDBSCAN, label clusters with representative entries —
   emergent themes without a fixed topic list, complementing the LLM ``enrich``
   topics.

   def discover_themes(min_cluster_size: int = 8) -> "pd.DataFrame":
       '''Cluster chunk vectors into themes; return cluster -> exemplar entries.'''
       raise NotImplementedError

Both read existing data and would emit their results as signals (their own
namespace) or as a side table — keeping presentation a pure query, consistent
with the five-layer design in ANALYSIS.md.
"""

from __future__ import annotations

__all__: list[str] = []
