# ============================================================
# LLM action cache — SQLite-backed, keyed by obs/mask/env/model/prompt/policy/provider
#
# Table: llm_action_cache
#   PK: (obs_hash, mask_hash, env_version, model_id, prompt_version,
#         policy_type, llm_provider)
#   Values: action (INT), reasoning (TEXT), created_at (TEXT)
# ============================================================

from __future__ import annotations

import hashlib
import pathlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np

_DEFAULT_CACHE_DB = pathlib.Path("data/llm_cache.db")


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


@contextmanager
def _connect(db_path: pathlib.Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_cache(db_path: pathlib.Path) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_action_cache (
                obs_hash      TEXT NOT NULL,
                mask_hash     TEXT NOT NULL,
                env_version   TEXT NOT NULL,
                model_id      TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                policy_type   TEXT NOT NULL,
                llm_provider  TEXT NOT NULL DEFAULT 'anthropic',
                action        INTEGER NOT NULL,
                reasoning     TEXT,
                created_at    TEXT NOT NULL,
                PRIMARY KEY (obs_hash, mask_hash, env_version, model_id,
                             prompt_version, policy_type, llm_provider)
            )
        """)
        # Migration: add llm_provider column to existing DBs
        try:
            conn.execute(
                "ALTER TABLE llm_action_cache ADD COLUMN llm_provider TEXT NOT NULL DEFAULT 'anthropic'"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists


def cache_lookup(
    obs: np.ndarray,
    mask: np.ndarray,
    env_version: str,
    model_id: str,
    prompt_version: str,
    policy_type: str,
    llm_provider: str = "anthropic",
    db_path: pathlib.Path = _DEFAULT_CACHE_DB,
) -> Optional[tuple[int, Optional[str]]]:
    """Look up cached action for the given key.

    Returns:
        (action, reasoning) if found, None otherwise.
    """
    _init_cache(db_path)
    obs_hash = _sha256(obs)
    mask_hash = _sha256(mask.astype(np.uint8))
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT action, reasoning FROM llm_action_cache
               WHERE obs_hash=? AND mask_hash=? AND env_version=?
                 AND model_id=? AND prompt_version=? AND policy_type=?
                 AND llm_provider=?""",
            (obs_hash, mask_hash, env_version, model_id, prompt_version,
             policy_type, llm_provider),
        ).fetchone()
    if row is None:
        return None
    return int(row["action"]), row["reasoning"]


def cache_store(
    obs: np.ndarray,
    mask: np.ndarray,
    env_version: str,
    model_id: str,
    prompt_version: str,
    policy_type: str,
    action: int,
    llm_provider: str = "anthropic",
    reasoning: Optional[str] = None,
    db_path: pathlib.Path = _DEFAULT_CACHE_DB,
) -> None:
    """Store an LLM action in the cache."""
    _init_cache(db_path)
    obs_hash = _sha256(obs)
    mask_hash = _sha256(mask.astype(np.uint8))
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO llm_action_cache
               (obs_hash, mask_hash, env_version, model_id, prompt_version,
                policy_type, llm_provider, action, reasoning, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs_hash, mask_hash, env_version, model_id, prompt_version,
                policy_type, llm_provider, action, reasoning,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
