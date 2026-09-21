"""Small, explicit SQLite persistence layer for analysis experiences."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from contextlib import contextmanager


class AnalysisPersistenceError(RuntimeError):
    """Raised when an analysis cannot be persisted or read."""


def strategy_fingerprint(strategy: dict[str, Any]) -> str:
    canonical = json.dumps(strategy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AnalysisStore:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        configured = path or os.getenv("NEXA_FUNDS_ANALYSIS_DB")
        self.path = Path(configured) if configured else Path(__file__).resolve().parent / "analysis.sqlite3"
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    strategy_json TEXT NOT NULL,
                    strategy_fingerprint TEXT NOT NULL,
                    backtest_metrics_json TEXT NOT NULL,
                    optimization_metrics_json TEXT NOT NULL,
                    walk_forward_metrics_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def insert(self, prompt: str, result: dict[str, Any]) -> dict[str, Any]:
        strategy = result.get("strategy")
        if not isinstance(strategy, dict):
            raise AnalysisPersistenceError("Analysis result has no serializable strategy.")
        try:
            strategy_json = json.dumps(strategy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            backtest = result.get("backtest", {})
            optimization = result.get("optimization", {})
            walk_forward = result.get("walk_forward", {})
            payloads = [json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                        for value in (backtest, optimization, walk_forward)]
        except (TypeError, ValueError) as exc:
            raise AnalysisPersistenceError(f"Analysis result is not JSON serializable: {exc}") from exc
        created_at = datetime.now(timezone.utc).isoformat()
        fingerprint = strategy_fingerprint(strategy)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses
                (prompt, strategy_json, strategy_fingerprint, backtest_metrics_json,
                 optimization_metrics_json, walk_forward_metrics_json, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(prompt), strategy_json, fingerprint, payloads[0], payloads[1], payloads[2],
                 str(result.get("summary") or ""), created_at),
            )
            connection.commit()
            record_id = cursor.lastrowid
        return self.get(record_id)

    def get(self, record_id: int) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise AnalysisPersistenceError(f"Analysis record {record_id} was not found.")
        return self._row_to_record(row)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1 or limit > 100:
            raise AnalysisPersistenceError("History limit must be between 1 and 100.")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            return {
                "id": row["id"],
                "prompt": row["prompt"],
                "strategy": json.loads(row["strategy_json"]),
                "strategy_fingerprint": row["strategy_fingerprint"],
                "backtest": json.loads(row["backtest_metrics_json"]),
                "optimization": json.loads(row["optimization_metrics_json"]),
                "walk_forward": json.loads(row["walk_forward_metrics_json"]),
                "summary": row["summary"],
                "created_at": row["created_at"],
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AnalysisPersistenceError(f"Stored analysis record is invalid: {exc}") from exc
