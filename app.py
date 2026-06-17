#!/usr/bin/env python3
"""Streamlit web UI for the local journal engine.

Three areas, all behind a single-password gate (Streamlit-native, no extra auth
dependency): Add entries, Analysis dashboard, Query (RAG chat).

The password is read from JOURNAL_PASSWORD (via .env / environment) and compared
with hmac.compare_digest. If it is unset the app refuses to start. Binds to
0.0.0.0 on a configurable port so the Mac can reach it over Tailscale — keep it
tailnet-only; never expose it publicly.
"""

from __future__ import annotations

import hmac
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# config.load_dotenv() runs on import, populating JOURNAL_PASSWORD et al.
from journal import config, ingest as ingest_mod, store
from journal.enrich import enrich, pending_entries
from journal.rag import ask
from journal.stats import (
    entries_per_year,
    has_enrichment,
    load_frame,
    mean_mood_per_year,
    top_tokens,
)

st.set_page_config(page_title="Journal Engine", page_icon="📓", layout="wide")


# --------------------------------------------------------------------------- #
# Auth gate (Streamlit session_state + constant-time compare)
# --------------------------------------------------------------------------- #
def require_auth() -> None:
    password = os.environ.get("JOURNAL_PASSWORD")
    if not password:
        st.error(
            "**JOURNAL_PASSWORD is not set — refusing to start.**\n\n"
            "This web UI exposes the most personal data you own, so it will not "
            "run without a password. Set it in a gitignored `.env` file "
            "(`JOURNAL_PASSWORD=...`) or export it in the environment, then "
            "reload."
        )
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("📓 Journal Engine")
    st.caption("Local-only. Tailnet-only. Enter the password to continue.")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if hmac.compare_digest(attempt, password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_auth()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_table():
    return store.open_or_create()


def summary_to_df(summary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_id": r.entry_id,
                "status": r.status,
                "date": r.date,
                "date_source": r.date_source,
                "chunks": r.n_chunks,
            }
            for r in summary.results
        ]
    )


def show_ingest_result(summary) -> None:
    st.success(
        f"{summary.n_entries} entries: +{summary.added} added, "
        f"~{summary.updated} updated, {summary.skipped} unchanged."
    )
    df = summary_to_df(summary)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    frac = summary.mtime_fraction()
    if frac > config.MTIME_WARN_FRACTION:
        st.warning(
            f"{frac:.0%} of new/changed entries fell back to file mtime for their "
            "date. Spot-check those before trusting temporal queries."
        )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📓 Journal Engine")
    tbl = get_table()
    st.metric("Chunks indexed", store.count_rows(tbl))
    st.caption(f"Embed: `{config.EMBED_MODEL}`  •  Chat: `{config.CHAT_MODEL}`")
    st.caption(f"DB: `{config.DB_PATH}`")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.rerun()

tab_add, tab_dash, tab_query = st.tabs(["➕ Add entries", "📊 Analysis", "💬 Query"])


# --------------------------------------------------------------------------- #
# 1. Add entries
# --------------------------------------------------------------------------- #
with tab_add:
    st.subheader("Add entries")
    mode = st.radio(
        "How are you adding?",
        ["Single entry", "Batch paste", "Upload .txt files"],
        horizontal=True,
    )

    if mode == "Single entry":
        with st.form("single"):
            date_str = st.text_input(
                "Date (optional — inferred from the text/header if blank)",
                placeholder="2019-07-14",
            )
            body = st.text_area("Entry text", height=240)
            go = st.form_submit_button("Ingest")
        if go and body.strip():
            summary = ingest_mod.ingest_paste(
                body, explicit_date=date_str or None, tbl=get_table()
            )
            show_ingest_result(summary)
        elif go:
            st.warning("Nothing to ingest — the entry is empty.")

    elif mode == "Batch paste":
        st.caption(
            "Paste multiple entries separated by a line of `---` / `===`, or by "
            "two or more blank lines. Each entry's date is inferred from its own "
            "header line, else today."
        )
        with st.form("batch"):
            blob = st.text_area("Entries", height=300)
            go = st.form_submit_button("Ingest batch")
        if go and blob.strip():
            summary = ingest_mod.ingest_batch(blob, tbl=get_table())
            show_ingest_result(summary)
        elif go:
            st.warning("Nothing to ingest.")

    else:  # Upload
        uploads = st.file_uploader(
            "Upload one or more .txt files",
            type=["txt"],
            accept_multiple_files=True,
        )
        if st.button("Ingest uploads") and uploads:
            files = []
            for uf in uploads:
                raw = uf.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1", errors="replace")
                files.append((uf.name, text))
            summary = ingest_mod.ingest_uploads(files, tbl=get_table())
            show_ingest_result(summary)
        elif st.session_state.get("_upload_clicked"):
            st.warning("No files selected.")


# --------------------------------------------------------------------------- #
# 2. Analysis dashboard
# --------------------------------------------------------------------------- #
with tab_dash:
    st.subheader("Analysis dashboard")

    tbl = get_table()
    pend = pending_entries(tbl)
    cols = st.columns([3, 1])
    with cols[1]:
        st.metric("Entries awaiting enrichment", pend)
        enrich_limit = st.number_input("Max entries per run", 1, 100000, 200)
        if st.button("Run / refresh enrichment", disabled=pend == 0):
            bar = st.progress(0.0, text="Enriching…")

            def _progress(n, total):
                bar.progress(n / total, text=f"Enriching {n}/{total} entries…")

            n = enrich(limit=int(enrich_limit), tbl=tbl, progress=_progress)
            bar.empty()
            st.success(f"Enriched {n} entries.")
            st.rerun()

    full = load_frame(tbl)
    if full.empty:
        st.info("No entries yet. Add some on the **Add entries** tab.")
    else:
        years = sorted(full["year"].dropna().unique())
        with cols[0]:
            if len(years) >= 2:
                lo, hi = st.select_slider(
                    "Date range (by year)",
                    options=years,
                    value=(years[0], years[-1]),
                )
            else:
                lo = hi = years[0] if years else None
        date_from = f"{lo}-01-01" if lo else None
        date_to = f"{hi}-12-31" if hi else None

        entries = load_frame(tbl, date_from=date_from, date_to=date_to)
        st.caption(f"{len(entries)} entries in range.")

        c1, c2 = st.columns(2)
        with c1:
            epy = entries_per_year(entries)
            if not epy.empty:
                fig = px.bar(
                    x=epy.index, y=epy.values,
                    labels={"x": "Year", "y": "Entries"},
                    title="Entries per year",
                )
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if has_enrichment(entries):
                mood = mean_mood_per_year(entries)
                if not mood.empty:
                    fig = px.line(
                        x=mood.index, y=mood.values, markers=True,
                        labels={"x": "Year", "y": "Mean mood (1-5)"},
                        title="Mean mood per year",
                    )
                    fig.update_yaxes(range=[1, 5])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run enrichment to see mood and people/place/topic charts.")

        if has_enrichment(entries):
            t1, t2, t3 = st.columns(3)
            for col_obj, (label, column) in zip(
                (t1, t2, t3),
                (("Top people", "people"), ("Top places", "places"),
                 ("Top topics", "topics")),
            ):
                with col_obj:
                    s = top_tokens(entries, column)
                    st.markdown(f"**{label}**")
                    if s.empty:
                        st.caption("(none)")
                    else:
                        fig = px.bar(
                            x=s.values, y=s.index, orientation="h",
                            labels={"x": "Mentions", "y": ""},
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 3. Query (RAG chat)
# --------------------------------------------------------------------------- #
with tab_query:
    st.subheader("Query your journal")
    qc = st.columns(3)
    with qc[0]:
        q_from = st.text_input("From (optional)", placeholder="2019-01-01")
    with qc[1]:
        q_to = st.text_input("To (optional)", placeholder="2019-12-31")
    with qc[2]:
        q_k = st.number_input("Excerpts to retrieve", 1, 30, 8)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("excerpts"):
                with st.expander("Cited excerpts"):
                    for h in msg["excerpts"]:
                        st.markdown(f"**[{h['date']}]** · `{h['entry_id']}`")
                        st.text(h["text"][:1200])

    prompt = st.chat_input("Ask about your journal…")
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning…"):
                ans = ask(
                    prompt, k=int(q_k),
                    date_from=q_from or None, date_to=q_to or None,
                    tbl=get_table(),
                )
            body = ans.text
            if ans.cited_dates:
                body += f"\n\n*— entries dated: {', '.join(ans.cited_dates)}*"
            st.markdown(body)
            if ans.excerpts:
                with st.expander("Cited excerpts"):
                    for h in ans.excerpts:
                        st.markdown(f"**[{h['date']}]** · `{h['entry_id']}`")
                        st.text(h["text"][:1200])
        st.session_state["chat"].append(
            {"role": "assistant", "content": body, "excerpts": ans.excerpts}
        )
