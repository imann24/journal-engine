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

import os
from datetime import date, datetime, timedelta

import extra_streamlit_components as stx
import pandas as pd
import plotly.express as px
import streamlit as st

# config.load_dotenv() runs on import, populating JOURNAL_PASSWORD et al.
from journal import config, ingest as ingest_mod, llm, store, conversations
from journal.webauth import COOKIE_NAME, auth_token, verify_password, verify_token
from journal.enrich import enrich, pending_entries
from journal.rag import ask
from journal.stats import (
    entries_per_year,
    has_enrichment,
    insight_frame,
    load_frame,
    mean_mood_per_year,
    needs_totals,
    reflective_prompts,
    signal_totals,
    top_tokens,
    topic_mood,
    year_signal_matrix,
)

st.set_page_config(page_title="Journal Engine", page_icon="📓", layout="wide")

# Initialize conversations database
conversations.init_db()

# Custom CSS to override the primary accent color to a soft blue.
# This keeps automatic dark/light mode detection fully functional.
st.markdown(
    """
    <style>
    :root {
        --primary-color: #3b82f6;
        --st-primary-color: #3b82f6;
    }
    .stApp {
        --primary-color: #3b82f6;
        --st-primary-color: #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------------------------- #
# Auth gate — constant-time password check, persisted per-browser via a signed
# cookie so you only enter the password once. The cookie holds an HMAC token
# (never the password); changing the password (or JOURNAL_AUTH_SECRET)
# invalidates all existing cookies. Token leakage is bounded by Tailscale's
# transport encryption + the tailnet-only deployment.
# --------------------------------------------------------------------------- #
AUTH_COOKIE = COOKIE_NAME
COOKIE_DAYS = int(os.environ.get("JOURNAL_AUTH_DAYS", "30"))
MODEL_COOKIE = "journal_model"  # persists the chat-model choice per browser


def require_auth() -> stx.CookieManager:
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

    # Construct once per script run (the component renders once); require_auth
    # is itself called once per run, so this stays a single widget instance.
    cookies = stx.CookieManager(key="journal_cookies")

    # Fast path: already authenticated this session.
    if st.session_state.get("authenticated"):
        return cookies

    # The cookie component returns {} until it has mounted and reported the
    # browser's cookies (one round-trip). Probe once before deciding, so a
    # remembered browser doesn't flash the login form.
    if not st.session_state.get("_cookie_probed"):
        st.session_state["_cookie_probed"] = True
        st.caption("Loading…")
        st.stop()

    token = cookies.get(AUTH_COOKIE)
    if verify_token(token, password):
        st.session_state["authenticated"] = True
        return cookies

    st.title("📓 Journal Engine")
    st.caption("Local-only. Tailnet-only. Enter the password to continue.")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        remember = st.checkbox("Remember this browser", value=True)
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if verify_password(attempt, password):
            st.session_state["authenticated"] = True
            if remember:
                cookies.set(
                    AUTH_COOKIE, auth_token(password),
                    expires_at=datetime.now() + timedelta(days=COOKIE_DAYS),
                )
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


cookie_manager = require_auth()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def get_table():
    # Open a fresh handle each call. A LanceDB table handle is a version
    # snapshot — caching one across reruns would make reads stale after any
    # write (and reference dropped data files after a wipe). Opening is cheap
    # for an embedded DB, so we always read the latest committed state.
    return store.open_or_create()


@st.cache_data(ttl=30)
def available_models() -> list[str]:
    return llm.list_models()


def selected_model() -> str:
    return st.session_state.get("chat_model") or config.CHAT_MODEL


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
        st.dataframe(df, width="stretch", hide_index=True)
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

    # Chat model picker — any model available on the local Ollama server. The
    # choice is remembered per browser (persisted via the cookie store), and
    # falls back to the default if that model is no longer served by Ollama.
    models = available_models()
    if not models:
        models = [config.CHAT_MODEL]
        st.caption("⚠️ Couldn't reach Ollama to list models; using the default.")

    def _fallback() -> str:
        return config.CHAT_MODEL if config.CHAT_MODEL in models else models[0]

    saved = cookie_manager.get(MODEL_COOKIE)
    if "chat_model" not in st.session_state:
        # Restore the remembered model on a fresh session, if still available.
        st.session_state["chat_model"] = saved if saved in models else _fallback()
    if st.session_state["chat_model"] not in models:
        # The remembered/selected model disappeared from Ollama — fall back.
        st.session_state["chat_model"] = _fallback()

    idx = models.index(st.session_state["chat_model"])
    chosen = st.selectbox(
        "Chat model", options=models, index=idx,
        help="Used for RAG answers and enrichment. Pick any model Ollama serves. "
             "Remembered on this browser.",
    )
    st.session_state["chat_model"] = chosen
    if chosen != saved:  # persist only on change (one cookie write)
        cookie_manager.set(
            MODEL_COOKIE, chosen, key="set_model",
            expires_at=datetime.now() + timedelta(days=365),
        )

    if st.button("↻ Refresh model list"):
        available_models.clear()
        st.rerun()

    st.caption(f"Embed: `{config.EMBED_MODEL}`")
    st.caption(f"DB: `{config.DB_PATH}`")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        try:
            cookie_manager.delete(AUTH_COOKIE)
        except Exception:
            pass  # cookie may already be absent
        st.rerun()

tab_add, tab_manage, tab_dash, tab_query = st.tabs(["➕ Add entries", "📂 Manage entries", "📊 Analysis", "💬 Query"])


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
        if "upload_key" not in st.session_state:
            st.session_state["upload_key"] = 0
        uploads = st.file_uploader(
            "Upload one or more .txt files",
            type=["txt"],
            accept_multiple_files=True,
            key=f"upload_files_{st.session_state['upload_key']}",
        )
        if st.button("Ingest uploads"):
            if uploads:
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
                st.session_state["upload_key"] += 1
                st.rerun()
            else:
                st.warning("No files selected.")

# --------------------------------------------------------------------------- #
# 2. Manage entries
# --------------------------------------------------------------------------- #
with tab_manage:
    st.subheader("Manage and View Entries")

    tbl = get_table()
    try:
        df_all = store.table_to_df(tbl)
    except Exception:
        df_all = pd.DataFrame()

    if df_all.empty:
        st.info("No entries indexed yet. Go to the **Add entries** tab to add your first journal entry!")
    else:
        entries = store.list_entries(tbl)

        # Two-column layout: Left is the list/search/controls, Right is the details view
        col_list, col_detail = st.columns([2, 3])

        with col_list:
            st.markdown("### 📋 Entries List")
            search_query = st.text_input(
                "🔍 Search entries",
                placeholder="Type to filter by text, date, source...",
                key="manage_search",
            )

            filtered_entries = entries.copy()
            if search_query:
                # Find entry IDs that match text content or metadata fields
                matching_ids = df_all[df_all["text"].str.contains(search_query, case=False, na=False)]["entry_id"].unique()
                meta_match = (
                    entries["entry_id"].str.contains(search_query, case=False, na=False) |
                    entries["date"].str.contains(search_query, case=False, na=False) |
                    entries["source"].str.contains(search_query, case=False, na=False)
                )
                filtered_entries = entries[entries["entry_id"].isin(matching_ids) | meta_match]

            st.caption(f"Showing {len(filtered_entries)} of {len(entries)} entries")

            # Interactive selection dataframe
            event = st.dataframe(
                filtered_entries,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="entries_dataframe",
            )

            # Action controls in an expander
            st.markdown("---")
            with st.expander("🗑️ Entry Removal Tools", expanded=False):
                selected_rows = event.selection.get("rows", []) if event else []
                selected_entry_id = None
                if selected_rows:
                    selected_entry_id = filtered_entries.iloc[selected_rows[0]]["entry_id"]

                # Option 1: Remove currently selected
                st.markdown("**Delete Selected**")
                if st.button("Delete selected entry", disabled=not selected_entry_id, type="primary", key="del_sel_btn"):
                    store.delete_entries(tbl, [selected_entry_id])
                    st.toast(f"Deleted entry: {selected_entry_id}")
                    st.rerun()

                st.divider()

                # Option 2: Remove specific chosen entries
                st.markdown("**Remove by ID selection**")
                picked = st.multiselect(
                    "Select entries to remove",
                    options=entries["entry_id"].tolist(),
                    key="manage_multiselect",
                )
                if st.button("Remove selected by ID", disabled=not picked, key="manage_remove_picked"):
                    store.delete_entries(tbl, picked)
                    st.toast(f"Removed {len(picked)} entries.")
                    st.rerun()

                st.divider()

                # Option 3: Remove by date range
                st.markdown("**Remove by date range**")
                d1, d2 = st.columns(2)
                with d1:
                    rm_from = st.text_input("From (YYYY-MM-DD)", placeholder="2019-01-01", key="rm_from_m")
                with d2:
                    rm_to = st.text_input("To (YYYY-MM-DD)", placeholder="2019-12-31", key="rm_to_m")
                if st.button("Remove range", disabled=not (rm_from or rm_to), key="manage_remove_range"):
                    ids = store.entry_ids_in_range(tbl, rm_from or None, rm_to or None)
                    if ids:
                        store.delete_entries(tbl, ids)
                        st.toast(f"Removed {len(ids)} entries in date range.")
                        st.rerun()
                    else:
                        st.warning("No entries found in that range.")

                st.divider()

                # Option 4: Delete ALL
                st.markdown("**Danger Zone**")
                confirm_all = st.checkbox("I'm sure — delete everything", key="manage_confirm_all")
                if st.button("Remove ALL entries", disabled=not confirm_all, key="manage_remove_all"):
                    store.delete_all(tbl)
                    st.toast("Database cleared successfully.")
                    st.rerun()

        with col_detail:
            selected_rows = event.selection.get("rows", []) if event else []
            if selected_rows:
                sel_row = filtered_entries.iloc[selected_rows[0]]
                entry_id = sel_row["entry_id"]

                # Fetch complete content and meta
                entry_chunks = df_all[df_all["entry_id"] == entry_id].sort_values("chunk_index")
                full_text = "\n".join(entry_chunks["text"].tolist())

                first_chunk = entry_chunks.iloc[0]
                date = first_chunk["date"]
                date_source = first_chunk["date_source"]
                source = first_chunk["source"]
                enriched = first_chunk.get("enriched", False)
                mood = first_chunk.get("mood", 0)
                topics = first_chunk.get("topics", "")
                places = first_chunk.get("places", "")
                people = first_chunk.get("people", "")

                st.markdown(f"### 📖 Entry Details")
                st.markdown(f"#### `{entry_id}`")

                # Metadata metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Date", date, help=f"Source: {date_source}")
                m2.metric("Source", os.path.basename(source) if "/" in source or "\\" in source else source)
                total_words = int(entry_chunks["word_count"].sum())
                m3.metric("Length", f"{total_words} words", help=f"Stored in {len(entry_chunks)} database chunks")

                st.divider()

                # AI Enrichment details if available
                if enriched:
                    st.markdown("#### ✨ AI Insights")

                    mood_map = {
                        1: "😢 Very Low",
                        2: "🙁 Low",
                        3: "😐 Neutral",
                        4: "🙂 High",
                        5: "😀 Very High"
                    }
                    mood_str = mood_map.get(mood, f"Unknown ({mood})")

                    # Columns for mood and entities
                    e1, e2 = st.columns([1, 2])
                    with e1:
                        st.metric("Mood Rating", mood_str)

                    with e2:
                        def parse_list(val):
                            if not val or val.strip().lower() in ("none", "null", "[]"):
                                return []
                            return [v.strip() for v in val.split(",") if v.strip()]

                        topics_list = parse_list(topics)
                        people_list = parse_list(people)
                        places_list = parse_list(places)

                        if topics_list:
                            st.markdown(f"**Topics:** {', '.join([f'`{t}`' for t in topics_list])}")
                        if people_list:
                            st.markdown(f"**People:** {', '.join([f'`{p}`' for p in people_list])}")
                        if places_list:
                            st.markdown(f"**Places:** {', '.join([f'`{p}`' for p in places_list])}")

                        if not (topics_list or people_list or places_list):
                            st.caption("No specific topics, people, or places identified.")

                    st.divider()
                else:
                    st.info("💡 This entry hasn't been enriched yet. Run enrichment in the **Analysis** tab to extract mood, topics, people, and places.")
                    st.divider()

                # Full entry text
                st.markdown("#### 📝 Content")
                st.text_area(
                    "Read-only View",
                    value=full_text,
                    height=450,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.markdown("### 📖 Entry Details")
                st.info("👈 Select an entry from the list on the left to view its full content, metadata, and AI enrichment insights.")


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

            n = enrich(limit=int(enrich_limit), tbl=tbl, progress=_progress,
                       model=selected_model())
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
                st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
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
                        st.plotly_chart(fig, width="stretch")

        insights = insight_frame(entries)
        if not insights.empty:
            st.divider()
            st.markdown("**Mindfulness signals**")
            st.caption(
                "Transparent lexical patterns for reflection, not diagnosis. "
                "Use them as gentle trailheads into the entries."
            )

            signals = signal_totals(insights)
            needs = needs_totals(insights)
            lens = insights["attention_lens"].value_counts()
            dominant_lens = lens.index[0] if not lens.empty else "reflective"
            dominant_signal = signals.index[0] if not signals.empty else "quiet"
            top_need = needs.index[0] if not needs.empty else "space"

            m1, m2, m3 = st.columns(3)
            m1.metric("Strongest signal", dominant_signal)
            m2.metric("Attention lens", dominant_lens)
            m3.metric("Recurring need", top_need)

            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Signal mix**")
                if signals.empty:
                    st.caption("Not enough signal words in this range yet.")
                else:
                    fig = px.bar(
                        x=signals.values, y=signals.index, orientation="h",
                        labels={"x": "Mentions", "y": ""},
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, width="stretch")
            with s2:
                st.markdown("**Needs and values**")
                if needs.empty:
                    st.caption("No recurring needs surfaced in this range yet.")
                else:
                    fig = px.bar(
                        x=needs.values, y=needs.index, orientation="h",
                        labels={"x": "Mentions", "y": ""},
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, width="stretch")

            matrix = year_signal_matrix(insights)
            if not matrix.empty and len(matrix) > 1:
                fig = px.imshow(
                    matrix.T,
                    aspect="auto",
                    labels={"x": "Year", "y": "Signal", "color": "Per 100 words"},
                    title="Signal weather by year",
                )
                st.plotly_chart(fig, width="stretch")

            tm = topic_mood(entries)
            if not tm.empty:
                st.markdown("**Topic mood companions**")
                fig = px.scatter(
                    tm.head(20),
                    x="entries", y="mean_mood", text="topic",
                    labels={
                        "entries": "Entries",
                        "mean_mood": "Mean mood",
                        "topic": "Topic",
                    },
                )
                fig.update_yaxes(range=[1, 5])
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, width="stretch")

            prompts = reflective_prompts(entries, insights)
            if prompts:
                st.markdown("**Reflection prompts**")
                for prompt in prompts:
                    st.write(f"- {prompt}")


# --------------------------------------------------------------------------- #
# 3. Query (RAG chat)
# --------------------------------------------------------------------------- #
def date_from_string(d_str: str | None) -> date | None:
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d").date()
    except Exception:
        return None

with tab_query:
    st.subheader("Query your journal")

    # Initialize session state variables for conversations
    if "active_conv_id" not in st.session_state:
        st.session_state["active_conv_id"] = None
    if "persist_active" not in st.session_state:
        st.session_state["persist_active"] = False
    if "q_from_date" not in st.session_state:
        st.session_state["q_from_date"] = None
    if "q_to_date" not in st.session_state:
        st.session_state["q_to_date"] = None
    if "q_k" not in st.session_state:
        st.session_state["q_k"] = 8
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Two-column layout: Left = Saved Chats history list, Right = Active Chat window & settings
    col_history, col_chat = st.columns([1, 3])

    with col_history:
        st.write("### 💬 Saved Chats")
        if st.button("➕ New Chat", width="stretch"):
            st.session_state["active_conv_id"] = None
            st.session_state["chat"] = []
            st.session_state["q_from_date"] = None
            st.session_state["q_to_date"] = None
            st.session_state["q_k"] = 8
            st.session_state["persist_active"] = False
            st.rerun()

        st.divider()

        # List saved conversations
        saved_convs = conversations.list_conversations()
        active_id = st.session_state["active_conv_id"]

        if not saved_convs:
            st.caption("No saved conversations yet.")
        else:
            for c in saved_convs:
                is_active = (c["id"] == active_id)
                btn_label = f"💬 {c['title']}" if is_active else c["title"]
                if st.button(
                    btn_label,
                    key=f"select_conv_{c['id']}",
                    width="stretch",
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state["active_conv_id"] = c["id"]
                    loaded = conversations.get_conversation(c["id"])
                    if loaded:
                        st.session_state["chat"] = loaded["messages"]
                        st.session_state["q_from_date"] = date_from_string(loaded["date_from"])
                        st.session_state["q_to_date"] = date_from_string(loaded["date_to"])
                        st.session_state["q_k"] = loaded["k"]
                        st.session_state["persist_active"] = True
                    st.rerun()

        if active_id:
            st.divider()
            st.write("### ⚙️ Chat Settings")
            active_conv = next((c for c in saved_convs if c["id"] == active_id), None)
            if active_conv:
                # Rename text input
                new_title = st.text_input("Rename title", value=active_conv["title"], key="rename_title_input")
                if new_title and new_title != active_conv["title"]:
                    conversations.rename_conversation(active_id, new_title)
                    st.toast("Conversation renamed!")
                    st.rerun()

                # Delete button
                if st.button("🗑️ Delete conversation", type="primary", width="stretch"):
                    conversations.delete_conversation(active_id)
                    st.session_state["active_conv_id"] = None
                    st.session_state["chat"] = []
                    st.session_state["q_from_date"] = None
                    st.session_state["q_to_date"] = None
                    st.session_state["q_k"] = 8
                    st.session_state["persist_active"] = False
                    st.toast("Conversation deleted.")
                    st.rerun()

    with col_chat:
        st.write("### 🔍 Query Options")
        qc = st.columns(3)
        with qc[0]:
            q_from_date = st.date_input("From (optional)", value=st.session_state["q_from_date"])
            st.session_state["q_from_date"] = q_from_date
            q_from = q_from_date.isoformat() if q_from_date else None
        with qc[1]:
            q_to_date = st.date_input("To (optional)", value=st.session_state["q_to_date"])
            st.session_state["q_to_date"] = q_to_date
            q_to = q_to_date.isoformat() if q_to_date else None
        with qc[2]:
            q_k = st.number_input("Excerpts to retrieve", 1, 30, value=st.session_state["q_k"])
            st.session_state["q_k"] = q_k

        active_id = st.session_state["active_conv_id"]

        # Option to persist/save this conversation
        if active_id:
            st.info("💾 This conversation is currently being saved to the persistent store.")
            persist_active = True
        else:
            persist_active = st.checkbox(
                "💾 Save this conversation",
                value=st.session_state.get("persist_active", False),
                help="If checked, this chat and its future queries will be stored locally so you can revisit them later."
            )
            st.session_state["persist_active"] = persist_active

        st.divider()

        # Display conversation messages
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

            # If checked and not yet saved in SQLite, initialize it
            if st.session_state["persist_active"]:
                if not active_id:
                    import uuid
                    active_id = str(uuid.uuid4())
                    st.session_state["active_conv_id"] = active_id
                    title = prompt[:40] + ("..." if len(prompt) > 40 else "")
                else:
                    conv_info = conversations.get_conversation(active_id)
                    title = conv_info["title"] if conv_info else (prompt[:40] + ("..." if len(prompt) > 40 else ""))

                conversations.save_conversation(
                    conversation_id=active_id,
                    title=title,
                    date_from=q_from,
                    date_to=q_to,
                    k=int(q_k),
                    messages=st.session_state["chat"]
                )

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and reasoning…"):
                    ans = ask(
                        prompt, k=int(q_k),
                        date_from=q_from or None, date_to=q_to or None,
                        tbl=get_table(), model=selected_model(),
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

            # Save the new response if persisting is enabled
            if st.session_state["persist_active"]:
                conv_info = conversations.get_conversation(active_id)
                title = conv_info["title"] if conv_info else (prompt[:40] + ("..." if len(prompt) > 40 else ""))
                conversations.save_conversation(
                    conversation_id=active_id,
                    title=title,
                    date_from=q_from,
                    date_to=q_to,
                    k=int(q_k),
                    messages=st.session_state["chat"]
                )

            st.rerun()
