"""Cross-process request pacing for shared academic API quotas."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path


RATE_LIMIT_DB_ENV = "PAPEREVAL_S2_GLOBAL_LIMIT_DB"


def wait_for_shared_slot(group: str, min_interval_seconds: float) -> None:
    """Reserve a request slot shared by all worker processes in this batch."""
    database = os.environ.get(RATE_LIMIT_DB_ENV)
    if not database or min_interval_seconds <= 0:
        return

    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS request_slots "
            "(group_name TEXT PRIMARY KEY, next_allowed REAL NOT NULL)"
        )
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        row = connection.execute(
            "SELECT next_allowed FROM request_slots WHERE group_name = ?", (group,)
        ).fetchone()
        slot = max(now, float(row[0]) if row else now)
        connection.execute(
            "INSERT INTO request_slots(group_name, next_allowed) VALUES (?, ?) "
            "ON CONFLICT(group_name) DO UPDATE SET next_allowed=excluded.next_allowed",
            (group, slot + min_interval_seconds),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

    wait = slot - time.time()
    if wait > 0:
        time.sleep(wait)
