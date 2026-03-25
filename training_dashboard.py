"""RL Training Dashboard — visualize training progress and reward heatmaps.

Usage:
    streamlit run training_dashboard.py
"""

import sqlite3
import os
import time

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

DB_PATH = os.path.join("rl_checkpoints", "training.db")
REFRESH_SECONDS = 10


def get_conn():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found: {DB_PATH}")
        st.stop()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@st.cache_data(ttl=REFRESH_SECONDS)
def load_episode_stats():
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT flag_id, success, steps, reward_total, tiles_explored, created_at
           FROM episode_stats ORDER BY created_at ASC""",
        conn,
    )
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["created_at"], unit="s")
    return df


@st.cache_data(ttl=REFRESH_SECONDS)
def load_tile_rewards(map_name: str):
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT x, y, reward, value, map_name
           FROM tile_rewards WHERE map_name = ?""",
        conn,
        params=(map_name,),
    )
    conn.close()
    return df


def _table_exists(conn, table_name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cur.fetchone() is not None


@st.cache_data(ttl=REFRESH_SECONDS)
def get_map_names():
    conn = get_conn()
    if not _table_exists(conn, "tile_rewards"):
        conn.close()
        return []
    df = pd.read_sql_query(
        "SELECT DISTINCT map_name FROM tile_rewards ORDER BY map_name", conn
    )
    conn.close()
    return df["map_name"].tolist()


# --- Page config ---
st.set_page_config(page_title="RL Training Dashboard", layout="wide")
st.title("RL Training Dashboard")

# Auto-refresh
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
if auto_refresh:
    time.sleep(0.1)  # avoid tight loop on first load
    st.sidebar.caption(f"Refreshing every {REFRESH_SECONDS}s")

# --- Load data ---
episodes = load_episode_stats()

if episodes.empty:
    st.warning("No episode data yet. Start training to see metrics.")
    if auto_refresh:
        time.sleep(REFRESH_SECONDS)
        st.rerun()
    st.stop()

# --- Key metrics ---
col1, col2, col3, col4 = st.columns(4)
recent = episodes.tail(50)
col1.metric("Total Episodes", len(episodes))
col2.metric("Recent Success Rate", f"{recent['success'].mean():.0%}")
col3.metric("Avg Reward (last 50)", f"{recent['reward_total'].mean():.1f}")
col4.metric("Avg Steps (last 50)", f"{recent['steps'].mean():.0f}")

# --- Episode reward over time ---
st.subheader("Episode Reward Over Time")
episodes["episode_num"] = range(1, len(episodes) + 1)
episodes["reward_rolling"] = episodes["reward_total"].rolling(20, min_periods=1).mean()

reward_chart = alt.Chart(episodes).mark_line(opacity=0.3, color="steelblue").encode(
    x=alt.X("episode_num:Q", title="Episode"),
    y=alt.Y("reward_total:Q", title="Reward"),
)
rolling_chart = alt.Chart(episodes).mark_line(color="orange", strokeWidth=2).encode(
    x="episode_num:Q",
    y=alt.Y("reward_rolling:Q", title="Reward (20-ep rolling avg)"),
)
st.altair_chart(reward_chart + rolling_chart, width="stretch")

# --- Success rate per flag ---
st.subheader("Success Rate by Flag")
flag_stats = (
    episodes.groupby("flag_id")
    .agg(total=("success", "count"), successes=("success", "sum"))
    .reset_index()
)
flag_stats["success_rate"] = flag_stats["successes"] / flag_stats["total"]

flag_chart = alt.Chart(flag_stats).mark_bar().encode(
    x=alt.X("flag_id:N", title="Flag", sort="-y"),
    y=alt.Y("success_rate:Q", title="Success Rate", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color(
        "success_rate:Q",
        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
        legend=None,
    ),
    tooltip=["flag_id", "success_rate", "total"],
)
st.altair_chart(flag_chart, width="stretch")

# --- Reward heatmap ---
st.subheader("Reward Heatmap by Map")

map_names = get_map_names()
if not map_names:
    st.info("No tile reward data yet.")
else:
    selected_map = st.selectbox("Select map", map_names)
    tile_df = load_tile_rewards(selected_map)

    if tile_df.empty:
        st.info("No tile data for this map.")
    else:
        view_mode = st.radio("View", ["Learned Value", "Raw Reward"], horizontal=True)

        # Aggregate per tile
        heatmap_df = tile_df.groupby(["x", "y"]).agg(
            avg_reward=("reward", "mean"),
            avg_value=("value", "mean"),
            visits=("reward", "count"),
        ).reset_index()

        if view_mode == "Learned Value":
            color_field = "avg_value:Q"
            color_title = "Avg Value"
        else:
            color_field = "avg_reward:Q"
            color_title = "Avg Reward"

        heat = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X("x:O", title="X"),
            y=alt.Y("y:O", title="Y", sort="ascending"),
            color=alt.Color(
                color_field,
                scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                title=color_title,
            ),
            tooltip=["x", "y", "avg_reward", "avg_value", "visits"],
        ).properties(
            width=600,
            height=500,
        )
        st.altair_chart(heat, width="stretch")

        st.caption(f"{len(heatmap_df)} unique tiles, {len(tile_df)} total visits")

# --- Auto-refresh ---
if auto_refresh:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
