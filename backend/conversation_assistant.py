"""Deterministic conversational layer for the analysis API."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.engine.strategy_analysis.analysis_service import StrategyAnalysisService


class ConversationStore:
    """Small SQLite profile and bounded message history store."""

    def __init__(self, path: str | os.PathLike[str] | None = None, max_messages: int = 20):
        configured = path or os.getenv("NEXA_FUNDS_ANALYSIS_DB")
        self.path = Path(configured) if configured else Path(__file__).resolve().parent / "analysis.sqlite3"
        self.max_messages = max_messages
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS user_profiles (user_id TEXT PRIMARY KEY, name TEXT, updated_at TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, role TEXT NOT NULL,
                message_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
            db.commit()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.path))
        try:
            yield connection
        finally:
            connection.close()

    def profile(self, user_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT name FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
            messages = db.execute(
                "SELECT role, message_json FROM conversation_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, self.max_messages),
            ).fetchall()
        return {"name": row[0] if row else None, "messages": [
            {"role": role, **json.loads(payload)} for role, payload in reversed(messages)
        ]}

    def save(self, user_id: str, role: str, message: str, **metadata: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            if metadata.get("name"):
                db.execute(
                    "INSERT INTO user_profiles(user_id,name,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
                    (user_id, metadata["name"], now),
                )
            db.execute("INSERT INTO conversation_messages(user_id,role,message_json,created_at) VALUES(?,?,?,?)",
                       (user_id, role, json.dumps({"message": message, **metadata}), now))
            db.execute(
                "DELETE FROM conversation_messages WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM conversation_messages WHERE user_id=? ORDER BY id DESC LIMIT ?)",
                (user_id, user_id, self.max_messages),
            )
            db.commit()


class IntentClassifier:
    NAME = re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z '-]{1,40})", re.I)
    # Keep this deliberately conservative: ordinary words such as "want" and
    # "best" must never become a fabricated market symbol.
    SYMBOL = re.compile(r"\b([A-Z]{6,}(?:[._-][A-Z0-9]+)?|[A-Z]{2,5}\d{1,4})\b", re.I)
    TIMEFRAME = re.compile(r"\b(M1|M5|M15|M30|H1|H4|D1|1m|5m|15m|30m|1h|4h|daily)\b", re.I)
    RISK = re.compile(r"\b(?:risk|risk per trade|drawdown)\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?", re.I)
    PLATFORM = re.compile(r"\b(MT4|MT5|MetaTrader\s*[45]|TradingView|cTrader)\b", re.I)

    def classify(self, message: str) -> tuple[str, dict[str, Any]]:
        text = message.strip()
        lower = text.lower()
        match = self.NAME.search(text)
        entities = {"name": match.group(1).strip(" .,!") if match else None}
        if re.search(r"\b(hello|hi|hey|how are you|good morning|good evening)\b", lower):
            return "greeting", entities
        if re.search(r"\b(best|profitable|indicator combinations?|combine indicators?|build an ea|ea platform)\b", lower):
            entities.update(self._constraints(text))
            entities["all_downloaded_data"] = bool(re.search(
                r"\ball\s+(?:downloaded|available)\s+(?:symbols?|markets?|data|datasets?)\b"
                r"|\ball\s+(?:downloaded|available)\s+data\b", lower))
            if entities["all_downloaded_data"]:
                entities["symbol"] = None
                entities["timeframe"] = None
            return "combination_search", entities
        if re.search(r"\b(strategy|backtest|buy|sell|entry|stop loss|take profit|analy[sz]e)\b", lower):
            return "strategy_analysis", entities
        return "casual", entities

    def _constraints(self, text: str) -> dict[str, Any]:
        symbol_values = [match.group(1).upper() for match in self.SYMBOL.finditer(text)]
        timeframe = self.TIMEFRAME.search(text)
        risk = self.RISK.search(text)
        platform = self.PLATFORM.search(text)
        ignored = {
            "PROFITABLE", "INDICATOR", "COMBINATIONS", "COMBINATION", "SEARCH",
            "DOWNLOADED", "AVAILABLE", "SYMBOLS", "MARKETS", "DATA", "DATASETS",
        }
        symbol_value = next((value for value in symbol_values if value not in ignored), None)
        return {
            "symbol": symbol_value,
            "timeframe": self._normalize_timeframe(timeframe.group(1)) if timeframe else None,
            "risk_percent": float(risk.group(1)) if risk else None,
            "platform": platform.group(1).upper().replace("METATRADER", "MT") if platform else None,
        }

    @staticmethod
    def _normalize_timeframe(value: str) -> str:
        value = value.upper()
        return {"1M": "M1", "5M": "M5", "15M": "M15", "30M": "M30", "1H": "H1", "4H": "H4", "DAILY": "D1"}.get(value, value)


COMBINATIONS = (
    ("EMA + RSI", "EMA_20 crosses above EMA_50 and RSI > 50"),
    ("MACD + ADX", "MACD > 0 and ADX > 20"),
    ("Bollinger Bands + RSI", "RSI < 30 and CLOSE crosses below BB_LOWER"),
)


class ConversationalAssistant:
    def __init__(self, store: ConversationStore | None = None, analysis_service: StrategyAnalysisService | None = None):
        self.store = store or ConversationStore()
        self.analysis_service = analysis_service or StrategyAnalysisService()
        self.classifier = IntentClassifier()

    def respond(self, message: str, user_id: str = "anonymous", progress_callback=None) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message is required.")
        intent, entities = self.classifier.classify(message)
        profile = self.store.profile(user_id)
        if not entities.get("name"):
            entities["name"] = profile.get("name")
        if intent == "greeting":
            name = entities.get("name")
            answer = f"Hello{', ' + name if name else ''}! I'm ready to help with measured strategy research."
            result = {"intent": intent, "message": answer, "profile": {"name": name}}
        elif intent == "combination_search":
            result = self._combination_search(message, entities, progress_callback=progress_callback)
        elif intent == "strategy_analysis":
            result = {"intent": intent, "message": "Please provide a symbol, timeframe, entry rules, risk, and historical data path for a measured analysis."}
        else:
            result = {"intent": intent, "message": "I can remember your name and help with evidence-based strategy research. What would you like to explore?"}
        self.store.save(user_id, "user", message, name=entities.get("name"))
        self.store.save(user_id, "assistant", result["message"], intent=intent)
        return result

    def _combination_search(self, message: str, entities: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        required = {
            "symbol": "Which symbol should be tested (for example XAUUSD)?",
            "timeframe": "Which timeframe should be tested (for example M15)?",
            "risk_percent": "What risk percentage per trade should be used?",
            "platform": "Which EA platform should the specification target (MT4, MT5, cTrader, or TradingView)?",
        }
        data_dir = Path(os.getenv("NEXA_FUNDS_DATA_DIR", Path(__file__).resolve().parent / "data"))
        if entities.get("all_downloaded_data"):
            return self._all_downloaded_search(entities, data_dir, progress_callback=progress_callback)

        missing = [key for key in required if not entities.get(key)]
        if entities.get("symbol") and entities.get("timeframe"):
            pattern = f"{entities['symbol']}_{entities['timeframe']}.csv"
            normalized_pattern = f"{entities['symbol']}_m_{entities['timeframe']}.csv"
            # Downloaded files have existed in both SYMBOL_TIMEFRAME and
            # SYMBOL_m_TIMEFRAME forms.  Keep the fast single-symbol path,
            # while accepting either spelling (and either configured root).
            candidates = (
                data_dir / pattern,
                data_dir / normalized_pattern,
                Path(__file__).resolve().parents[1] / "data" / pattern,
                Path(__file__).resolve().parents[1] / "data" / normalized_pattern,
            )
            if not any(candidate.exists() for candidate in candidates):
                missing.append("data")
        else:
            missing.append("data")
        if missing:
            return {"intent": "combination_search", "status": "needs_clarification",
                    "questions": [{"field": field, "question": required.get(field, "Please provide a CSV with timestamp, open, high, low, close, and volume columns.")} for field in dict.fromkeys(missing)],
                    "message": "I need these constraints before testing combinations; I won't fabricate a profitable recommendation."}
        candidates = []
        for name, rules in COMBINATIONS:
            prompt = f"BUY {entities['symbol']} on {entities['timeframe']} when {rules}. risk {entities['risk_percent']}%"
            report = self.analysis_service.analyze(prompt)
            if report.get("backtest", {}).get("status") == "completed" and report.get("walk_forward", {}).get("status") == "completed":
                candidates.append({"name": name, "strategy": report.get("strategy"), "backtest": report["backtest"], "walk_forward": report["walk_forward"], "ea_platform": entities["platform"]})
        return {"intent": "combination_search", "status": "completed", "candidates": candidates,
                "message": f"Measured walk-forward results found {len(candidates)} candidate(s). Results are historical evidence, not a guarantee of profitability."}

    def _all_downloaded_search(self, entities: dict[str, Any], data_dir: Path, progress_callback=None) -> dict[str, Any]:
        """Run the bounded combination library over every valid downloaded dataset."""
        required = {
            "risk_percent": "What risk percentage per trade should be used?",
            "platform": "Which EA platform should the specification target (MT4, MT5, cTrader, or TradingView)?",
        }
        missing = [key for key in required if not entities.get(key)]
        if missing:
            return {
                "intent": "combination_search",
                "status": "needs_clarification",
                "scope": "all_downloaded_data",
                "questions": [{"field": key, "question": required[key]} for key in missing],
                "message": "I need the risk and platform before testing all downloaded datasets; I won't fabricate a recommendation.",
            }

        max_datasets = self._positive_limit("NEXA_FUNDS_MAX_DATASETS", 25)
        datasets, skipped = self._discover_datasets(data_dir)
        selected = datasets[:max_datasets]
        if len(datasets) > max_datasets:
            skipped.append({
                "file": None,
                "reason": f"Dataset limit reached ({max_datasets}); {len(datasets) - max_datasets} valid dataset(s) not run.",
            })

        per_dataset = []
        records = []
        combinations = COMBINATIONS[:self._positive_limit("NEXA_FUNDS_MAX_COMBINATIONS", len(COMBINATIONS))]
        if progress_callback:
            progress_callback(total=len(selected) * len(combinations), completed=0, current_dataset=None, current_combination=None)
        completed = 0
        for dataset in selected:
            symbol, timeframe = dataset["symbol"], dataset["timeframe"]
            item = {**dataset, "status": "completed", "candidates": []}
            for name, rules in combinations:
                if progress_callback:
                    progress_callback(total=len(selected) * len(combinations), completed=completed,
                                      current_dataset=f"{symbol}_{timeframe}", current_combination=name)
                prompt = f"BUY {symbol} on {timeframe} when {rules}. risk {entities['risk_percent']}%"
                try:
                    report = self.analysis_service.analyze(prompt)
                except Exception as exc:
                    item["status"] = "error"
                    item.setdefault("errors", []).append({"combination": name, "reason": str(exc)})
                    completed += 1
                    if progress_callback:
                        progress_callback(total=len(selected) * len(combinations), completed=completed,
                                          current_dataset=f"{symbol}_{timeframe}", current_combination=name)
                    continue
                measured = (
                    report.get("backtest", {}).get("status") == "completed"
                    and report.get("walk_forward", {}).get("status") == "completed"
                )
                if not measured:
                    item.setdefault("skipped", []).append({
                        "combination": name,
                        "reason": report.get("backtest", {}).get("reason")
                        or "Measured backtest and completed walk-forward validation were not both available.",
                    })
                    completed += 1
                    if progress_callback:
                        progress_callback(total=len(selected) * len(combinations), completed=completed,
                                          current_dataset=f"{symbol}_{timeframe}", current_combination=name)
                    continue
                candidate = {
                    "name": name,
                    "strategy": report.get("strategy"),
                    "backtest": report["backtest"],
                    "walk_forward": report["walk_forward"],
                    "ea_platform": entities["platform"],
                }
                item["candidates"].append(candidate)
                records.append({
                    "id": f"{symbol}_{timeframe}_{name}",
                    "strategy": report.get("strategy") or {"symbol": symbol, "timeframe": timeframe},
                    "backtest": report["backtest"],
                    "walk_forward": report["walk_forward"],
                })
                completed += 1
                if progress_callback:
                    progress_callback(total=len(selected) * len(combinations), completed=completed,
                                      current_dataset=f"{symbol}_{timeframe}", current_combination=name)
            per_dataset.append(item)

        from backend.combination_ranking import CombinationRankingService
        aggregate = CombinationRankingService().rank(records, limit=10)
        return {
            "intent": "combination_search",
            "scope": "all_downloaded_data",
            "status": "completed",
            "datasets": per_dataset,
            "skipped_datasets": skipped,
            "aggregate_ranking": aggregate,
            "limits": {
                "max_datasets": max_datasets,
                "combinations_per_dataset": len(combinations),
                "discovered": len(datasets),
                "executed": len(selected),
            },
            "message": (
                f"Measured walk-forward results found {len(aggregate)} aggregate candidate(s) "
                f"across {len(selected)} dataset(s). Results are historical evidence, not a guarantee of profitability."
            ),
        }

    @staticmethod
    def _positive_limit(name: str, default: int) -> int:
        try:
            return max(1, min(int(os.getenv(name, str(default))), 1000))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _discover_datasets(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        supported = {"M1", "M5", "M15", "M30", "H1", "H4"}
        datasets, skipped = [], []
        roots = [data_dir]
        repository_root = Path(__file__).resolve().parents[1] / "data"
        # The bundled two-root layout is used when no custom test/data directory
        # is configured; custom directories remain isolated and deterministic.
        if data_dir.resolve() == (Path(__file__).resolve().parent / "data").resolve():
            roots.append(repository_root)
        paths = []
        for root in roots:
            if root.exists():
                paths.extend(root.glob("*.csv"))
            elif root == data_dir:
                skipped.append({"file": str(root), "reason": "Data directory does not exist."})
        seen = set()
        for path in sorted(paths, key=lambda item: str(item).lower()):
            stem = path.stem
            stem = re.sub(r"_m_", "_", stem, flags=re.I)
            match = re.match(r"^(.+)_([A-Za-z0-9]+)$", stem)
            if not match:
                skipped.append({"file": str(path), "reason": "Filename must be SYMBOL_TIMEFRAME.csv."})
                continue
            symbol, timeframe = match.group(1).upper(), match.group(2).upper()
            if not symbol or not re.match(r"^[A-Z0-9][A-Z0-9._-]*$", symbol):
                skipped.append({"file": str(path), "reason": "Malformed symbol in filename."})
                continue
            if timeframe not in supported:
                skipped.append({"file": str(path), "reason": f"Unsupported timeframe '{timeframe}'."})
                continue
            try:
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    columns = {column.strip().lower() for column in next(csv.reader(stream), [])}
            except (OSError, UnicodeError, csv.Error) as exc:
                skipped.append({"file": str(path), "reason": f"Could not read CSV: {exc}"})
                continue
            missing = sorted({"timestamp", "open", "high", "low", "close", "volume"} - columns)
            if missing:
                skipped.append({"file": str(path), "reason": f"Missing required OHLCV columns: {missing}"})
                continue
            try:
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                if not rows:
                    raise ValueError("CSV contains no data rows.")
                for row_number, row in enumerate(rows, start=2):
                    if not row.get("timestamp", "").strip():
                        raise ValueError(f"row {row_number} has an empty timestamp")
                    try:
                        datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00"))
                    except ValueError:
                        raise ValueError(f"row {row_number} has an invalid timestamp")
                    for column in ("open", "high", "low", "close", "volume"):
                        if not row.get(column, "").strip():
                            raise ValueError(f"row {row_number} has an empty {column}")
                        float(row[column])
            except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                skipped.append({"file": str(path), "reason": f"Malformed OHLCV data: {exc}"})
                continue
            key = (symbol, timeframe)
            if key in seen:
                skipped.append({"file": str(path), "reason": "Duplicate dataset; deterministic first file retained."})
                continue
            seen.add(key)
            datasets.append({"file": str(path), "symbol": symbol, "timeframe": timeframe})
        return datasets, skipped
