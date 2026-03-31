"""SQLite database for parallel RL training.

Stores experience tuples from multiple workers and episode statistics.
Thread-safe — SQLite handles concurrent reads, and we use WAL mode for
concurrent writes from multiple processes.
"""

import sqlite3
import os
import json
import time
import numpy as np
import torch

DB_PATH = os.path.join("rl_checkpoints", "training.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a database connection with WAL mode for concurrent access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: str = DB_PATH):
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL,
            flag_id TEXT NOT NULL,
            step_num INTEGER NOT NULL,
            state BLOB NOT NULL,
            action INTEGER NOT NULL,
            reward REAL NOT NULL,
            value REAL NOT NULL,
            log_prob REAL NOT NULL,
            done INTEGER NOT NULL,
            created_at REAL NOT NULL,
            consumed INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS episode_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL,
            flag_id TEXT NOT NULL,
            success INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            reward_total REAL NOT NULL,
            tiles_explored INTEGER NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tile_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            map_id INTEGER NOT NULL,
            map_name TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            reward REAL NOT NULL,
            value REAL NOT NULL DEFAULT 0.0,
            flag_id TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_experience_consumed
            ON experience(consumed, created_at);
        CREATE INDEX IF NOT EXISTS idx_episode_stats_flag
            ON episode_stats(flag_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_tile_rewards_map
            ON tile_rewards(map_id, x, y);
    """)
    conn.close()


def write_experience(
    conn: sqlite3.Connection,
    worker_id: int,
    flag_id: str,
    step_num: int,
    state: torch.Tensor,
    action: int,
    reward: float,
    value: float,
    log_prob: float,
    done: bool,
):
    """Write a single experience tuple to the database."""
    state_bytes = state.numpy().tobytes()
    conn.execute(
        """INSERT INTO experience
           (worker_id, flag_id, step_num, state, action, reward, value, log_prob, done, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (worker_id, flag_id, step_num, state_bytes, action, reward, value, log_prob,
         int(done), time.time()),
    )


def write_experience_batch(
    conn: sqlite3.Connection,
    worker_id: int,
    flag_id: str,
    experiences: list[tuple],
):
    """Write a batch of experience tuples. Each tuple: (step, state, action, reward, value, logprob, done)."""
    now = time.time()
    rows = []
    for step, state, action, reward, value, log_prob, done in experiences:
        state_bytes = state.numpy().tobytes()
        rows.append((worker_id, flag_id, step, state_bytes, action, reward, value,
                      log_prob, int(done), now))
    conn.executemany(
        """INSERT INTO experience
           (worker_id, flag_id, step_num, state, action, reward, value, log_prob, done, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def read_unconsumed_experience(
    conn: sqlite3.Connection,
    state_size: int,
    limit: int = 4096,
) -> tuple[list[int], list[dict]]:
    """Read unconsumed experience for PPO update.

    Returns:
        (ids, experiences) where experiences are dicts with state tensor, action, reward, etc.
    """
    cursor = conn.execute(
        """SELECT id, state, action, reward, value, log_prob, done
           FROM experience WHERE consumed = 0
           ORDER BY created_at ASC LIMIT ?""",
        (limit,),
    )
    ids = []
    experiences = []
    for row in cursor:
        exp_id, state_bytes, action, reward, value, log_prob, done = row
        ids.append(exp_id)
        state_arr = np.frombuffer(state_bytes, dtype=np.float32).copy()
        if state_arr.size != state_size:
            # Skip stale experience from an older observation schema.
            continue
        state = torch.from_numpy(state_arr)
        experiences.append({
            "state": state,
            "action": action,
            "reward": reward,
            "value": value,
            "log_prob": log_prob,
            "done": bool(done),
        })
    return ids, experiences


def mark_consumed(conn: sqlite3.Connection, ids: list[int]):
    """Mark experience rows as consumed after PPO update."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"UPDATE experience SET consumed = 1 WHERE id IN ({placeholders})", ids)
    conn.commit()


def write_episode_stats(
    conn: sqlite3.Connection,
    worker_id: int,
    flag_id: str,
    success: bool,
    steps: int,
    reward_total: float,
    tiles_explored: int,
):
    """Record episode results."""
    conn.execute(
        """INSERT INTO episode_stats
           (worker_id, flag_id, success, steps, reward_total, tiles_explored, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (worker_id, flag_id, int(success), steps, reward_total, tiles_explored, time.time()),
    )
    conn.commit()


def get_recent_episode_stats(
    conn: sqlite3.Connection,
    flag_id: str,
    limit: int = 20,
) -> list[dict]:
    """Get recent episode stats for a flag (across all workers)."""
    cursor = conn.execute(
        """SELECT success, steps, reward_total, tiles_explored
           FROM episode_stats WHERE flag_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (flag_id, limit),
    )
    return [
        {"success": bool(row[0]), "steps": row[1], "reward_total": row[2], "tiles_explored": row[3]}
        for row in cursor
    ]


def count_unconsumed(conn: sqlite3.Connection) -> int:
    """Count unconsumed experience rows."""
    cursor = conn.execute("SELECT COUNT(*) FROM experience WHERE consumed = 0")
    return cursor.fetchone()[0]


def write_tile_reward(
    conn: sqlite3.Connection,
    map_id: int,
    map_name: str,
    x: int,
    y: int,
    reward: float,
    flag_id: str,
):
    """Record reward earned at a specific tile."""
    conn.execute(
        """INSERT INTO tile_rewards (map_id, map_name, x, y, reward, flag_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (map_id, map_name, x, y, reward, flag_id, time.time()),
    )


def write_tile_rewards_batch(
    conn: sqlite3.Connection,
    rows: list[tuple],
):
    """Write a batch of tile reward rows. Each tuple: (map_id, map_name, x, y, reward, value, flag_id)."""
    now = time.time()
    conn.executemany(
        """INSERT INTO tile_rewards (map_id, map_name, x, y, reward, value, flag_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(m, mn, x, y, r, v, f, now) for m, mn, x, y, r, v, f in rows],
    )
    conn.commit()


def cleanup_old_experience(conn: sqlite3.Connection, max_age_hours: float = 24.0):
    """Delete consumed experience older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    conn.execute("DELETE FROM experience WHERE consumed = 1 AND created_at < ?", (cutoff,))
    conn.commit()
