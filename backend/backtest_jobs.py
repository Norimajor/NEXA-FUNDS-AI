"""Threaded background jobs for bounded all-dataset backtests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
from typing import Any

from backend.conversation_assistant import ConversationalAssistant


class BacktestJobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="backtest")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def submit(self, message: str, user_id: str) -> dict[str, Any]:
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        state = {
            "job_id": job_id, "status": "queued", "progress": 0.0,
            "current_dataset": None, "current_combination": None,
            "completed": 0, "total": 0, "elapsed_seconds": 0.0,
            "eta_seconds": None, "estimated_remaining_seconds": None,
            "result": None, "error": None,
            "_started": now, "_finished": None,
        }
        with self._lock:
            self._jobs[job_id] = state
        self._executor.submit(self._run, job_id, message, user_id)
        return self.status(job_id)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.update(values)
            elapsed = (datetime.now(timezone.utc) - state["_started"]).total_seconds()
            state["elapsed_seconds"] = round(max(0.0, elapsed), 3)
            if state["total"]:
                state["progress"] = round(min(1.0, state["completed"] / state["total"]), 4)
                if state["completed"] and state["completed"] < state["total"]:
                    eta = round(
                        elapsed * (state["total"] - state["completed"]) / state["completed"], 3
                    )
                    state["eta_seconds"] = eta
                    state["estimated_remaining_seconds"] = eta

    def _run(self, job_id: str, message: str, user_id: str) -> None:
        self._update(job_id, status="running")

        def progress(**values: Any) -> None:
            self._update(job_id, **values)

        try:
            result = ConversationalAssistant().respond(message, user_id, progress_callback=progress)
            with self._lock:
                total = self._jobs[job_id]["total"]
            self._update(job_id, status="completed", result=result, completed=total)
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc))
        finally:
            with self._lock:
                state = self._jobs.get(job_id)
                if state:
                    state["_finished"] = datetime.now(timezone.utc)
                    state["elapsed_seconds"] = round((state["_finished"] - state["_started"]).total_seconds(), 3)
                    if state["status"] == "completed":
                        state["progress"] = 1.0
                        state["eta_seconds"] = 0.0
                        state["estimated_remaining_seconds"] = 0.0

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                raise KeyError(job_id)
            return {key: value for key, value in state.items() if not key.startswith("_")}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._jobs)
        return [self.status(job_id) for job_id in ids]


backtest_jobs = BacktestJobManager()
