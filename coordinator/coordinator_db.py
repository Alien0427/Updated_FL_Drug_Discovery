"""Coordinator database helpers for idempotent update processing."""

import sqlite3
from datetime import datetime
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "ledger" / "ledger.db"


def _resolve_db_path(db_path: str) -> str:
    """Resolve a database path relative to the project ledger directory.

    Args:
        db_path: Caller-provided database path or the shorthand ``ledger.db``.

    Returns:
        Absolute filesystem path to the SQLite database file.
    """
    if db_path == "ledger.db":
        return str(_DEFAULT_DB_PATH)
    return db_path


def check_if_duplicate(update_id: str, db_path: str = "ledger.db") -> bool:
    """Check whether an update has already been committed to the ledger.

    Args:
        update_id: Unique client response identifier.
        db_path: SQLite database path or ``ledger.db`` shorthand.

    Returns:
        True if a committed row with the same ``update_id`` already exists.
    """
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        cursor = conn.execute(
            """
            SELECT update_id
            FROM checkpoint_ledger
            WHERE update_id = ?
              AND status = 'update_committed'
            """,
            (update_id,),
        )
        row = cursor.fetchone()

    return row is not None


def log_to_ledger(
    query_id: str,
    client_id: str,
    update_id: str,
    status: str,
    db_path: str = "ledger.db",
) -> None:
    """Append a checkpoint event to the SQLite ledger.

    Duplicate replay events are stored as separate audit rows so the original
    committed update remains immutable.

    Args:
        query_id: Federated query identifier assigned by the coordinator.
        client_id: Responding client identifier.
        update_id: Unique client update or response identifier.
        status: Ledger status such as ``update_committed`` or ``duplicate_ignored``.
        db_path: SQLite database path or ``ledger.db`` shorthand.

    Returns:
        None
    """
    resolved_path = _resolve_db_path(db_path)
    timestamp = datetime.now().isoformat()
    ledger_update_id = update_id

    if status == "duplicate_ignored":
        ledger_update_id = f"{update_id}::duplicate_ignored::{query_id}"

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        try:
            conn.execute(
                """
                INSERT INTO checkpoint_ledger (
                    query_id,
                    client_id,
                    update_id,
                    status,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (query_id, client_id, ledger_update_id, status, timestamp),
            )
        except sqlite3.IntegrityError:
            return


def get_audit_trail(limit: int = 100, db_path: str = "ledger.db") -> list[dict]:
    """Fetch recent checkpoint ledger rows for the audit dashboard."""
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT id, query_id, client_id, update_id, status, timestamp
            FROM checkpoint_ledger
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_latest_client_checkpoint(client_name: str, db_path: str = "ledger.db") -> dict | None:
    """Fetch the most recent committed checkpoint for one federated client."""
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT update_id, timestamp
            FROM checkpoint_ledger
            WHERE client_id = ?
              AND status = 'update_committed'
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (client_name,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return dict(row)
