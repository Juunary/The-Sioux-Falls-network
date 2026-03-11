# ============================================================
# Experiment tracker — SQLite-backed storage for evaluation runs
#
# Schema:
#   experiments(id, experiment_id, policy, split, created_at, git_commit,
#               graph_version, scenario_count, config_json)
#   episode_metrics(id, experiment_id, scenario_id, policy,
#                   episode_reward, passengers_spawned, passengers_served,
#                   passengers_gone, service_rate, unserved_rate,
#                   avg_wait_time_sec, charge_events, total_distance_px,
#                   wall_time_sec)
# ============================================================

from __future__ import annotations

import json
import sqlite3
import subprocess
import pathlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from backend.datasets.evaluator import EpisodeMetrics

_DEFAULT_DB = pathlib.Path(__file__).parent.parent.parent / "data" / "experiments.db"


# ============================================================
# DB helpers
# ============================================================

def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@contextmanager
def _connect(db_path: pathlib.Path = _DEFAULT_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db(db_path: pathlib.Path = _DEFAULT_DB) -> None:
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                policy        TEXT NOT NULL,
                split         TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                git_commit    TEXT,
                graph_version TEXT DEFAULT 'v1',
                scenario_count INTEGER DEFAULT 0,
                config_json   TEXT
            );

            CREATE TABLE IF NOT EXISTS episode_metrics (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id    TEXT NOT NULL,
                scenario_id      TEXT NOT NULL,
                policy           TEXT NOT NULL,
                episode_reward   REAL,
                passengers_spawned INTEGER,
                passengers_served  INTEGER,
                passengers_gone    INTEGER,
                service_rate     REAL,
                unserved_rate    REAL,
                avg_wait_time_sec REAL,
                charge_events    INTEGER,
                total_distance_px REAL,
                wall_time_sec    REAL
            );
        """)


# ============================================================
# Public API
# ============================================================

def create_experiment(
    experiment_id: str,
    policy: str,
    split: str,
    config: Optional[dict] = None,
    db_path: pathlib.Path = _DEFAULT_DB,
) -> str:
    """Register a new experiment run. Returns experiment_id."""
    _init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO experiments
               (experiment_id, policy, split, created_at, git_commit, config_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                policy,
                split,
                datetime.now(tz=timezone.utc).isoformat(),
                _get_git_commit(),
                json.dumps(config) if config else None,
            ),
        )
    return experiment_id


def record_metrics(
    experiment_id: str,
    metrics: list[EpisodeMetrics],
    db_path: pathlib.Path = _DEFAULT_DB,
) -> None:
    """Insert episode metrics records and update experiment scenario count."""
    _init_db(db_path)
    with _connect(db_path) as conn:
        for m in metrics:
            d = asdict(m)
            conn.execute(
                """INSERT INTO episode_metrics
                   (experiment_id, scenario_id, policy, episode_reward,
                    passengers_spawned, passengers_served, passengers_gone,
                    service_rate, unserved_rate, avg_wait_time_sec,
                    charge_events, total_distance_px, wall_time_sec)
                   VALUES (:experiment_id, :scenario_id, :policy, :episode_reward,
                           :passengers_spawned, :passengers_served, :passengers_gone,
                           :service_rate, :unserved_rate, :avg_wait_time_sec,
                           :charge_events, :total_distance_px, :wall_time_sec)""",
                {**d, "experiment_id": experiment_id},
            )
        conn.execute(
            "UPDATE experiments SET scenario_count = ? WHERE experiment_id = ?",
            (len(metrics), experiment_id),
        )


def list_experiments(db_path: pathlib.Path = _DEFAULT_DB) -> list[dict]:
    """Return all experiment records."""
    _init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment_metrics(
    experiment_id: str,
    db_path: pathlib.Path = _DEFAULT_DB,
) -> list[dict]:
    """Return episode metrics for a given experiment."""
    _init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM episode_metrics WHERE experiment_id = ? ORDER BY scenario_id",
            (experiment_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def aggregate_metrics(experiment_id: str, db_path: pathlib.Path = _DEFAULT_DB) -> dict:
    """Return aggregate statistics for an experiment."""
    _init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT
               COUNT(*) as count,
               AVG(episode_reward) as avg_reward,
               AVG(service_rate) as avg_service_rate,
               AVG(unserved_rate) as avg_unserved_rate,
               AVG(avg_wait_time_sec) as avg_wait_time_sec,
               MIN(episode_reward) as min_reward,
               MAX(episode_reward) as max_reward
               FROM episode_metrics WHERE experiment_id = ?""",
            (experiment_id,),
        ).fetchone()
    return dict(row) if row else {}
