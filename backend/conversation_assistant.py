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
            columns = {row[1] for row in db.execute("PRAGMA table_info(user_profiles)")}
            if "onboarding_state" not in columns:
                db.execute("ALTER TABLE user_profiles ADD COLUMN onboarding_state TEXT NOT NULL DEFAULT 'new'")
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
            row = db.execute("SELECT name, onboarding_state FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
            messages = db.execute(
                "SELECT role, message_json FROM conversation_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, self.max_messages),
            ).fetchall()
        return {"name": row[0] if row else None, "onboarding_state": row[1] if row else "new", "messages": [
            {"role": role, **json.loads(payload)} for role, payload in reversed(messages)
        ]}

    def save(self, user_id: str, role: str, message: str, **metadata: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            if metadata.get("name") or metadata.get("onboarding_state"):
                db.execute(
                    "INSERT INTO user_profiles(user_id,name,onboarding_state,updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(user_id) DO UPDATE SET "
                    "name=COALESCE(excluded.name,user_profiles.name), "
                    "onboarding_state=COALESCE(excluded.onboarding_state,user_profiles.onboarding_state), "
                    "updated_at=excluded.updated_at",
                    (user_id, metadata.get("name"), metadata.get("onboarding_state", "new"), now),
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
    NAME = re.compile(
        r"\b(?:my name is|i am|i'm|you're speaking with|you are speaking with|speaking with)\s+"
        r"([A-Za-z][A-Za-z '-]{1,40})",
        re.I,
    )
    SYMBOL = re.compile(r"\b([A-Z]{6,}(?:[._-][A-Z0-9]+)?|[A-Z]{2,5}\d{1,4})\b", re.I)
    TIMEFRAME = re.compile(r"\b(M1|M5|M15|M30|H1|H4|D1|1m|5m|15m|30m|1h|4h|daily)\b", re.I)
    RISK = re.compile(r"\b(?:risk|risk per trade|drawdown)\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?", re.I)
    PLATFORM = re.compile(r"\b(MT4|MT5|MetaTrader\s*[45]|TradingView|cTrader)\b", re.I)
    INDICATORS = (
        "RSI", "EMA", "MACD", "ADX", "ATR", "STOCHASTIC", "BOLLINGER", "CCI",
        "WILLIAMS %R", "MFI", "VWAP", "OBV", "SAR", "ICHIMOKU", "DONCHIAN",
        "SUPERTREND", "SUPPLY", "DEMAND", "MOMENTUM", "ROC", "TREND",
    )

    def classify(self, message: str) -> tuple[str, dict[str, Any]]:
        text = message.strip()
        lower = text.lower()
        match = self.NAME.search(text)
        entities = {"name": match.group(1).strip(" .,!") if match else None}
        if re.fullmatch(r"\s*(yes|yeah|yep|sure|okay|ok|of course)\s*[.!]?\s*", lower):
            return "affirmative", entities
        if re.search(r"\b(hello|hi|hey|how are you|good morning|good evening)\b", lower):
            greeting = re.search(r"\b(hello|hi|hey)\b", lower)
            entities["greeting_word"] = greeting.group(1).capitalize() if greeting else "Hello"
            return "greeting", entities
        if re.search(r"\b(?:what\s+is|what\s+are|explain|describe|tell\s+me\s+about)\b", lower):
            if re.search(r"\b(?:rsi|macd|ema|adx|atr|bollinger|stochastic|cci|mfi|williams|vwap|ichimoku|supertrend|sar|supply|demand)\b", lower):
                return "indicator_explanation", entities
            return "education", entities
        if re.search(r"\b(?:test|backtest|run|analyze|evaluate)\b", lower) and (
            re.search(r"\b(?:it|that|this|strategy|previous|last)\b", lower)
            or re.search(r"\bstrategy\b", lower)
        ):
            return "backtest_request", entities
        if re.search(r"\b(?:why\s+did|what\s+caused|why\s+did\s+it|what\s+explains|why\s+did\s+it\s+perform|why\s+did\s+it\s+lose)\b", lower):
            return "backtest_analysis", entities
        if re.search(r"\b(?:optimize|improve|refine|tune|make\s+it\s+better|be\s+more\s+strict|make\s+it\s+stricter)\b", lower):
            return "optimization_request", entities
        if re.search(r"\b(?:change|modify|update|replace|remove|use|switch|make)\b", lower) and re.search(r"\b(?:ema|rsi|macd|adx|atr|stochastic|bollinger|cci|mfi|williams|vwap|ichimoku|supertrend|sar|indicator|trend)\b", lower):
            return "strategy_modification", entities
        if re.search(r"\b(?:build|create|design|draft|write|develop|construct)\b.*\b(?:strategy|system|setup)\b", lower):
            return "strategy_creation", entities
        if re.search(r"\b(?:go\s+long|go\s+short|buy\s+when|sell\s+when|long\s+when|short\s+when|when\s+rsi\s+is|ema\s+cross|price\s+above\s+the\s+\d+\s+ema|price\s+below\s+the\s+\d+\s+ema)\b", lower):
            return "strategy_creation", entities
        if re.search(r"\b(best|profitable|indicator combinations?|combine indicators?|build an ea|ea platform)\b", lower):
            entities.update(self._constraints(text))
            entities["all_downloaded_data"] = bool(re.search(
                r"\ball\s+(?:downloaded|available)\s+(?:symbols?|markets?|data|datasets?)\b"
                r"|\ball\s+(?:downloaded|available)\s+data\b", lower))
            if entities["all_downloaded_data"]:
                entities["symbol"] = None
                entities["timeframe"] = None
            return "combination_search", entities
        if re.search(r"\b(strategy|backtest|buy|sell|entry|stop loss|take profit|analy[sz]e|risk|timeframe|symbol)\b", lower):
            return "strategy_analysis", entities
        return "general_conversation", entities

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
        self._session_state: dict[str, dict[str, Any]] = {}

    def _session(self, user_id: str) -> dict[str, Any]:
        return self._session_state.setdefault(user_id, {
            "last_strategy_text": None,
            "last_strategy": None,
            "last_backtest": None,
            "last_summary": None,
            "last_intent": None,
            "current_topic": None,
        })

    def _resolve_reference(self, message: str, state: dict[str, Any]) -> str | None:
        lower = message.lower()
        if "test it" in lower or "test that" in lower or "backtest it" in lower or "run it" in lower or "try it again" in lower:
            return state.get("last_strategy_text")
        if "that strategy" in lower or "this strategy" in lower or "previous strategy" in lower or "the previous strategy" in lower:
            return state.get("last_strategy_text")
        if "previous test" in lower or "last test" in lower or "that test" in lower:
            return state.get("last_strategy_text")
        if re.search(r"\b(?:it|that|this)\b", lower) and state.get("last_strategy_text"):
            return state.get("last_strategy_text")
        return None

    def _modify_strategy_text(self, strategy_text: str, message: str) -> str:
        text = strategy_text
        lower = message.lower()
        if any(fragment in lower for fragment in ("ema 50", "ema fifty", "ema to 50", "use 50 ema", "make the ema 50", "change the ema to 50")):
            if re.search(r"\bEMA\s*[_-]?\d+\b", text, flags=re.I):
                text = re.sub(r"\bEMA\s*[_-]?\d+\b", "EMA 50", text, flags=re.I)
                text = re.sub(r"\bema\s*[_-]?\d+\b", "EMA 50", text, flags=re.I)
                return text
            if text.rstrip().endswith("."):
                text = text.rstrip(".")
            return f"{text} and EMA 50 trend filter."
        if any(fragment in lower for fragment in ("ema 200", "ema to 200", "change the ema to 200")):
            if re.search(r"\bEMA\s*[_-]?\d+\b", text, flags=re.I):
                text = re.sub(r"\bEMA\s*[_-]?\d+\b", "EMA 200", text, flags=re.I)
                text = re.sub(r"\bema\s*[_-]?\d+\b", "EMA 200", text, flags=re.I)
                return text
            if text.rstrip().endswith("."):
                text = text.rstrip(".")
            return f"{text} and EMA 200 trend filter."
        if "remove rsi" in lower or "drop rsi" in lower:
            text = re.sub(r"\b[^.]*RSI[^.]*[.;]?\s*", "", text, flags=re.I)
            return text.strip()
        if "remove macd" in lower or "drop macd" in lower:
            text = re.sub(r"\b[^.]*MACD[^.]*[.;]?\s*", "", text, flags=re.I)
            return text.strip()
        if re.search(r"\b(?:rsi\s+below\s+(\d+)|rsi\s+under\s+(\d+)|rsi\s+is\s+below\s+(\d+))\b", lower):
            match = re.search(r"\b(?:rsi\s+below\s+(\d+)|rsi\s+under\s+(\d+)|rsi\s+is\s+below\s+(\d+))\b", lower)
            threshold = next(group for group in match.groups() if group is not None)
            text = re.sub(r"RSI\s+(?:below|under|is\s+below)\s+\d+", f"RSI below {threshold}", text, flags=re.I)
            return text
        return text

    def _handle_backtest_request(self, message: str, user_id: str, state: dict[str, Any]) -> dict[str, Any]:
        strategy_text = self._resolve_reference(message, state) or message
        if self._is_vague_strategy_request(strategy_text):
            return {
                "intent": "clarification_needed",
                "message": "I need the symbol, timeframe, and entry rules before I can backtest it. If you want, say: 'XAUUSD M15 long when RSI is below 30.'",
            }
        try:
            report = self.analysis_service.analyze(strategy_text)
        except Exception as exc:
            return {
                "intent": "clarification_needed",
                "message": f"I couldn't turn that into a valid strategy: {exc}",
            }
        state["last_strategy_text"] = strategy_text
        state["last_strategy"] = report.get("strategy")
        state["last_backtest"] = report
        state["last_summary"] = report.get("summary")
        state["last_intent"] = "backtest_request"
        if not report.get("success"):
            return {
                "intent": "clarification_needed",
                "message": report.get("error") or "I need clearer rules before I can test it.",
            }
        metrics = report.get("backtest", {})
        trade_count = metrics.get("trades") if isinstance(metrics, dict) else None
        net_return = metrics.get("net_return") if isinstance(metrics, dict) else None
        symbol = (report.get("strategy") or {}).get("symbol") or (state.get("last_strategy") or {}).get("symbol") or "the instrument"
        summary = f"I tested {symbol} and the measured backtest result is: {metrics.get('status', 'completed')}."
        if trade_count is not None:
            summary += f" Trades: {trade_count}."
        if net_return is not None:
            summary += f" Net return: {net_return}."
        return {
            "intent": "backtest_request",
            "message": summary,
            "strategy": report.get("strategy"),
            "backtest": report.get("backtest"),
            "test": report.get("test"),
        }

    def _handle_backtest_analysis(self, message: str, user_id: str, state: dict[str, Any]) -> dict[str, Any]:
        last_backtest = state.get("last_backtest") or {}
        if not last_backtest:
            return {
                "intent": "backtest_analysis",
                "message": "I don't have a recent backtest result to analyze yet. Tell me which strategy to test or run a backtest first.",
            }
        metrics = last_backtest.get("backtest", {}) if isinstance(last_backtest, dict) else {}
        net_return = metrics.get("net_return")
        win_rate = metrics.get("win_rate")
        max_drawdown = metrics.get("max_drawdown")
        return {
            "intent": "backtest_analysis",
            "message": (
                f"The previous backtest showed net return {net_return} and max drawdown {max_drawdown}. "
                f"Win rate was {win_rate}. I would review the entry timing, risk sizing, and whether the strategy is overfitting the selected regime."
            ),
            "metrics": metrics,
        }

    def _handle_strategy_modification(self, message: str, user_id: str, state: dict[str, Any]) -> dict[str, Any]:
        strategy_text = self._resolve_reference(message, state) or state.get("last_strategy_text")
        if not strategy_text:
            return {
                "intent": "clarification_needed",
                "message": "I don't have a recent strategy in this conversation yet. Tell me the strategy you want to change.",
            }
        updated_text = self._modify_strategy_text(strategy_text, message)
        state["last_strategy_text"] = updated_text
        state["last_intent"] = "strategy_modification"
        return {
            "intent": "strategy_modification",
            "message": f"I updated the strategy to: {updated_text}. I can test that version next.",
            "strategy_text": updated_text,
        }

    def _is_vague_strategy_request(self, text: str) -> bool:
        lower = text.lower()
        return (
            "strategy" in lower and not re.search(r"\b(?:xauusd|eurusd|gbpusd|usdjpy|audusd|nzdusd|usdcad|usdchf|eurjpy|gbpjpy|audjpy|nas100|us30|btcusd|ethusd|gold)\b", lower)
            and not re.search(r"\b(?:m1|m5|m15|m30|h1|h4|15m|5m|30m|1h|4h)\b", lower)
        )

    def _explain_indicator(self, text: str) -> str:
        lower = text.lower()
        if "rsi" in lower:
            return "RSI measures the speed and magnitude of recent price changes. Readings below 30 often suggest oversold conditions, while readings above 70 often suggest overbought conditions."
        if "macd" in lower:
            return "MACD compares fast and slow moving averages to show momentum and trend changes. Crossovers are often used as directional signals."
        if "ema" in lower:
            return "The EMA gives more weight to recent prices than a simple average, making it more responsive to recent trend changes. A 50 or 200 EMA is common for trend filtering."
        if "atr" in lower:
            return "ATR measures average true range and is commonly used to estimate volatility and set stop-loss distance."
        if "stochastic" in lower:
            return "Stochastic compares the closing price to its recent range, and it is often used to spot momentum and potential reversal zones."
        if "bollinger" in lower:
            return "Bollinger Bands show a moving average with upper and lower bands based on volatility; price near the edges can signal exhaustion or expansion."
        return "I can explain trading concepts like RSI, MACD, EMAs, ATR, and momentum indicators. Tell me which one you want to understand."

    def respond(self, message: str, user_id: str = "anonymous", progress_callback=None) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message is required.")
        intent, entities = self.classifier.classify(message)
        profile = self.store.profile(user_id)
        state = self._session(user_id)
        if not entities.get("name"):
            entities["name"] = profile.get("name")
        if intent == "greeting":
            name = entities.get("name")
            if name:
                state["onboarding_state"] = "awaiting_offer_confirmation"
                answer = f"Welcome {name}! Wanna know how I can help you?"
                state["current_topic"] = "greeting"
                state["last_intent"] = intent
                result = {"intent": intent, "message": answer, "profile": {"name": name}}
            else:
                greeting = entities.get("greeting_word", "Hello")
                state["onboarding_state"] = "awaiting_name"
                answer = f"{greeting}, I am NEXAFUNDS AI. Who am I speaking with please?"
                state["current_topic"] = "greeting"
                state["last_intent"] = intent
                result = {"intent": intent, "message": answer, "profile": {"name": name}}
        elif intent == "affirmative":
            if profile.get("name") or state.get("onboarding_state") == "awaiting_offer_confirmation":
                result = {
                    "intent": "capabilities",
                    "message": (
                        "I can understand natural-language trading ideas, validate and backtest strategies "
                        "on downloaded market data, compare indicator combinations, run walk-forward validation, "
                        "show progress for multi-market searches, rank measured candidates, and prepare "
                        "EA-ready strategy specifications. I will always distinguish historical evidence "
                        "from guaranteed profitability. What would you like to build?"
                    ),
                }
                state["current_topic"] = "capabilities"
                state["last_intent"] = intent
                state["onboarding_state"] = "active"
            else:
                result = {"intent": "affirmative", "message": "Thanks. I can help with strategy design, validation, and measured backtests."}
                state["current_topic"] = "affirmative"
                state["last_intent"] = intent
        elif intent == "indicator_explanation":
            result = {"intent": intent, "message": self._explain_indicator(message)}
            state["current_topic"] = "indicator_explanation"
            state["last_intent"] = intent
        elif intent == "strategy_modification":
            result = self._handle_strategy_modification(message, user_id, state)
            state["current_topic"] = "strategy_modification"
        elif intent == "backtest_request":
            result = self._handle_backtest_request(message, user_id, state)
            state["current_topic"] = "backtest_request"
        elif intent == "backtest_analysis":
            result = self._handle_backtest_analysis(message, user_id, state)
            state["current_topic"] = "backtest_analysis"
        elif intent == "clarification_needed":
            result = {
                "intent": "clarification_needed",
                "message": "What timeframe and entry/exit rules do you want me to use for that strategy?",
            }
            state["current_topic"] = "clarification_needed"
        elif intent == "strategy_creation":
            if self._is_vague_strategy_request(message):
                result = {
                    "intent": "clarification_needed",
                    "message": "What timeframe and entry/exit rules should I use for that gold strategy?",
                }
            else:
                strategy_text = message
                report = self.analysis_service.analyze(strategy_text)
                state["last_strategy_text"] = strategy_text
                state["last_strategy"] = report.get("strategy")
                state["last_backtest"] = report
                state["last_summary"] = report.get("summary")
                state["last_intent"] = intent
                if not report.get("success"):
                    result = {"intent": "clarification_needed", "message": report.get("error") or "I need a clearer strategy before I can test it."}
                else:
                    result = {
                        "intent": "strategy_creation",
                        "message": f"I interpreted that as a strategy for {report.get('strategy', {}).get('symbol', 'the instrument')}. I can test it next.",
                        "strategy": report.get("strategy"),
                        "validation": report.get("validation"),
                    }
            state["current_topic"] = "strategy_creation"
        elif intent == "combination_search":
            result = self._combination_search(message, entities, progress_callback=progress_callback)
            state["current_topic"] = "combination_search"
            state["last_intent"] = intent
        elif intent == "strategy_analysis":
            result = {"intent": intent, "message": "Please provide a symbol, timeframe, entry rules, risk, and historical data path for a measured analysis."}
            state["current_topic"] = "strategy_analysis"
            state["last_intent"] = intent
        else:
            subject = message.strip()
            result = {"intent": intent, "message": "I can help with trading concepts, strategy design, and measured backtests. What would you like to explore?"}
            if "gold" in subject.lower() or "xauusd" in subject.lower():
                result["message"] = "I can help with gold and XAUUSD strategy questions. Tell me the timeframe and the exact setup you want to test."
            state["current_topic"] = intent
            state["last_intent"] = intent
        if entities.get("name") and not profile.get("name"):
            state["current_topic"] = "greeting"
            result = {
                "intent": "greeting",
                "message": f"Welcome {entities['name']}! Wanna know how I can help you?",
                "profile": {"name": entities["name"]},
            }
        self.store.save(user_id, "user", message, name=entities.get("name"), onboarding_state=state.get("onboarding_state", profile.get("onboarding_state", "active")))
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
