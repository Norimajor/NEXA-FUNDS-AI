from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.ml_consensus import h1_engine, ml_router, m15_engine, processor, verify_api_key
from core.trading_engine import TradingEngine
from market_data.deribit import DeribitOptionsClient
from market_data.gold_options import GoldOptionsConnector
from market_data.options import ExecutionAdapter, OptionsRiskManager, OptionsSignalEngine
from market_data.service import OptionsMarketDataService
from ml.predictor import ModelPredictor
from data.tick_database import MarketDatabase

from datetime import timedelta

load_dotenv()

app = FastAPI(
    title="NexaFunds Analysis AI",
    version="2.0.0",
    description="Supply/demand and intelligent trend-channel analysis service for NexaFunds.",
)
app.include_router(ml_router)

origins = [
    x.strip() for x in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    ).split(",") if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

db = MarketDatabase()
predictor = ModelPredictor(os.getenv("MODEL_PATH", "models/current.joblib"), float(os.getenv("MODEL_THRESHOLD", "0.62")))
engine = TradingEngine(predictor)
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")
options_service = OptionsMarketDataService(deribit_client=DeribitOptionsClient(), gold_connector=GoldOptionsConnector())
options_signal_engine = OptionsSignalEngine()
options_risk_manager = OptionsRiskManager({"stale_data_seconds": 300, "signal_max_age_seconds": 600})
execution_adapter = ExecutionAdapter(paper_trading=True, live_trading=False)


def _trend_predictor_for_timeframe(timeframe: str):
    normalized = timeframe.strip().upper()
    if normalized.startswith("PERIOD_"):
        normalized = normalized[len("PERIOD_"):]
    if normalized in {"H1", "1H", "60", "60M"}:
        return h1_engine.predictor
    if normalized in {"M15", "15M", "15"}:
        return m15_engine.predictor
    return None


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class CandleBatch(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    timeframe: str = Field(min_length=1, max_length=10)
    candles: list[Candle] = Field(min_length=1)


@app.post("/api/ml/predict")
def ml_predict_endpoint(batch: CandleBatch, request: Request, x_api_key: str | None = Header(default=None)):
    _verify_key(
        x_api_key,
        allow_loopback=True,
        client_host=request.client.host if request.client else None,
    )
    if len(batch.candles) < 260:
        raise HTTPException(status_code=400, detail="At least 260 completed candles are required for model inference.")

    try:
        prepared = processor.prepare([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in batch.candles
        ])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = engine.analyze(
        prepared, trend_predictor=_trend_predictor_for_timeframe(batch.timeframe)
    )
    signal = result["signal"]
    confidence = float(result.get("confidence", 0.0) or 0.0)
    buy_percent = float(result.get("probabilities", {}).get("buy", 0.0) or 0.0)
    sell_percent = float(result.get("probabilities", {}).get("sell", 0.0) or 0.0)
    buy_prob = buy_percent / 100.0
    sell_prob = sell_percent / 100.0
    levels = result.get("levels", {}) or {}
    bullish_reversal = levels.get("potential_bullish_reversal") or {}
    bearish_reversal = levels.get("potential_bearish_reversal") or {}
    model_version = result.get("model_version") or predictor.bundle.get("version", "unknown") if predictor.bundle else "NO_MODEL"

    payload = {
        "symbol": batch.symbol.upper(),
        "timeframe": batch.timeframe.upper(),
        "signal": signal,
        "signal_reason": result.get("signal_reason", "UNKNOWN"),
        "confidence": round(confidence, 2),
        "setup_type": result.get("setup_type", "unknown"),
        "direction": result.get("direction", "NONE"),
        "setup_probability": round(float(result.get("setup_probability", confidence / 100.0) or 0.0), 4),
        "buy": round(buy_percent, 2),
        "sell": round(sell_percent, 2),
        "buy_probability": round(buy_prob, 4),
        "sell_probability": round(sell_prob, 4),
        "probability_reversal": round(float(result.get("probabilities", {}).get("probability_reversal", 0.0)), 4),
        "probability_continuation": round(float(result.get("probabilities", {}).get("probability_continuation", 0.0)), 4),
        "probability_breakout": round(float(result.get("probabilities", {}).get("probability_breakout", 0.0)), 4),
        "probability_failed_breakout": round(float(result.get("probabilities", {}).get("probability_failed_breakout", 0.0)), 4),
        "probability_no_edge": round(float(result.get("probabilities", {}).get("probability_no_edge", 0.0)), 4),
        "probability_reversal_buy": round(float(result.get("probabilities", {}).get("probability_reversal_buy", 0.0)), 4),
        "probability_reversal_sell": round(float(result.get("probabilities", {}).get("probability_reversal_sell", 0.0)), 4),
        "probability_continuation_buy": round(float(result.get("probabilities", {}).get("probability_continuation_buy", 0.0)), 4),
        "probability_continuation_sell": round(float(result.get("probabilities", {}).get("probability_continuation_sell", 0.0)), 4),
        "probability_breakout_buy": round(float(result.get("probabilities", {}).get("probability_breakout_buy", 0.0)), 4),
        "probability_breakout_sell": round(float(result.get("probabilities", {}).get("probability_breakout_sell", 0.0)), 4),
        "probability_failed_breakout_buy": round(float(result.get("probabilities", {}).get("probability_failed_breakout_buy", 0.0)), 4),
        "probability_failed_breakout_sell": round(float(result.get("probabilities", {}).get("probability_failed_breakout_sell", 0.0)), 4),
        "probability_long": round(float(result.get("probabilities", {}).get("probability_long", buy_prob)), 4),
        "probability_short": round(float(result.get("probabilities", {}).get("probability_short", sell_prob)), 4),
        "probability_flat": round(float(result.get("probabilities", {}).get("probability_flat", max(0.0, 1.0 - max(buy_prob, sell_prob)))), 4),
        "level_source": levels.get("source", "structural_reference_not_model_price_prediction"),
        "potential_bullish_reversal_source": levels.get("potential_bullish_reversal_source", "unavailable"),
        "potential_bearish_reversal_source": levels.get("potential_bearish_reversal_source", "unavailable"),
        "potential_bullish_reversal_low": bullish_reversal.get("low", 0.0),
        "potential_bullish_reversal_high": bullish_reversal.get("high", 0.0),
        "potential_bullish_reversal_mid": bullish_reversal.get("mid", 0.0),
        "potential_bearish_reversal_low": bearish_reversal.get("low", 0.0),
        "potential_bearish_reversal_high": bearish_reversal.get("high", 0.0),
        "potential_bearish_reversal_mid": bearish_reversal.get("mid", 0.0),
        "breakout_up": levels.get("breakout_up") or 0.0,
        "breakout_down": levels.get("breakout_down") or 0.0,
        "model_version": model_version,
        "channel_bias": result.get("channel_bias", "NONE"),
        "timestamp": result.get("timestamp"),
        "price": result.get("price"),
        "risk": result.get("risk", {}),
        "features": result.get("features", {}),
    }
    return payload


def _verify_key(x_api_key: str | None, allow_loopback=False, client_host=None):
    verify_api_key(
        x_api_key,
        allow_loopback=allow_loopback,
        client_host=client_host,
        api_key=INGEST_API_KEY,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "nexafunds-analysis-ai",
        "version": "2.0.0",
        "models": {
            "h1": h1_engine.predictor.bundle.get("version", "unknown") if h1_engine.predictor.bundle else "NO_MODEL",
            "m15": m15_engine.predictor.bundle.get("version", "unknown") if m15_engine.predictor.bundle else "NO_MODEL",
        },
    }


@app.post("/api/market/candles")
def ingest_candles(batch: CandleBatch, x_api_key: str | None = Header(default=None)):
    _verify_key(x_api_key)
    count = db.insert_candles(
        batch.symbol, batch.timeframe, [c.model_dump() for c in batch.candles]
    )
    return {"status": "ok", "symbol": batch.symbol.upper(), "timeframe": batch.timeframe.upper(), "stored": count}


@app.get("/api/analysis/{symbol}/{timeframe}")
def analysis(symbol: str, timeframe: str, limit: int = 500):
    candles = db.fetch_candles(symbol, timeframe, limit)
    if candles.empty:
        raise HTTPException(status_code=404, detail="No market data found for this symbol/timeframe.")

    prepared = processor.prepare(candles)
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "analysis": _safe(engine.analyze(
            prepared, trend_predictor=_trend_predictor_for_timeframe(timeframe)
        )),
    }


@app.get("/api/channels/{symbol}/{timeframe}")
def channels(symbol: str, timeframe: str, limit: int = 500):
    candles = db.fetch_candles(symbol, timeframe, limit)
    if candles.empty:
        raise HTTPException(status_code=404, detail="No market data found.")
    prepared = processor.prepare(candles)
    result = engine.analyze(
        prepared, trend_predictor=_trend_predictor_for_timeframe(timeframe)
    )
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "primary_channel": _safe(result["primary_channel"]),
        "channels": _safe(result["channels"]),
        "price": result["price"],
        "channel_bias": result["channel_bias"],
    }


@app.get("/api/supply-demand/{symbol}/{timeframe}")
def supply_demand(symbol: str, timeframe: str, limit: int = 500):
    candles = db.fetch_candles(symbol, timeframe, limit)
    if candles.empty:
        raise HTTPException(status_code=404, detail="No market data found.")
    prepared = processor.prepare(candles)
    result = engine.analyze(
        prepared, trend_predictor=_trend_predictor_for_timeframe(timeframe)
    )
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "zones": _safe(result["supply_demand_zones"]),
        "nearest_demand": _safe(result["nearest_demand"]),
        "nearest_supply": _safe(result["nearest_supply"]),
    }


def _btc_reference_price(contracts: list[Any]) -> float | None:
    valid = [float(contract.underlying_price) for contract in contracts if getattr(contract, "underlying_price", None) is not None]
    return max(valid) if valid else None


def _snapshot_age_seconds(contracts: list[Any]) -> float:
    timestamps = [contract.timestamp for contract in contracts if getattr(contract, "timestamp", None) is not None]
    if not timestamps:
        return float("inf")
    latest = max(timestamps)
    age = datetime.now(timezone.utc) - latest
    return max(age.total_seconds(), 0.0)


def build_btc_signal_response() -> dict[str, Any]:
    snapshot = options_service.get_btc_snapshot("BTC")
    contracts = snapshot.get("contracts") or []
    signal = snapshot.get("signal") or {"signal": "HOLD", "regime": "INSUFFICIENT_DATA", "confidence": 0.0, "score": 0.0, "data_quality": {"sufficient": False, "missing_fields": []}, "reasons": ["No valid BTC signal available."]}
    if not contracts:
        return {
            "asset": "BTC",
            "signal": "HOLD",
            "regime": "INSUFFICIENT_DATA",
            "score": 0.0,
            "confidence": 0.0,
            "signal_id": "btc-empty",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc)).isoformat(),
            "entry_reference": None,
            "features": snapshot.get("features", {}),
            "risk": snapshot.get("risk", {"allowed": False, "reason": "NO_CONTRACTS", "reasons": ["Deribit returned no option contracts."]}),
            "data_quality": snapshot.get("data_quality", {"sufficient": False, "reason": "NO_CONTRACTS", "missing_fields": ["contracts"]}),
            "execution": {"paper_trading": True, "live_trading": False, "mode": "PAPER"},
            "reasons": ["No valid BTC option contracts were returned by Deribit."],
        }

    stale_seconds = _snapshot_age_seconds(contracts)
    if stale_seconds > options_risk_manager.config.get("stale_data_seconds", 300):
        return {
            "asset": "BTC",
            "signal": "HOLD",
            "regime": "STALE_DATA",
            "score": 0.0,
            "confidence": 0.0,
            "signal_id": "btc-stale",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc)).isoformat(),
            "entry_reference": _btc_reference_price(contracts),
            "features": snapshot.get("features", {}) or options_signal_engine.feature_engine.build(contracts, asset="BTC", reference_price=_btc_reference_price(contracts)).to_dict(),
            "risk": {"allowed": False, "reason": "STALE_DATA", "reasons": ["BTC option snapshot is older than the safety window."]},
            "data_quality": {"sufficient": False, "reason": "STALE_DATA", "missing_fields": ["timestamp"]},
            "execution": {"paper_trading": True, "live_trading": False, "mode": "PAPER"},
            "reasons": ["Options snapshot is stale and rejected by the safety gate."],
        }

    risk = snapshot.get("risk", {"allowed": False, "reason": "UNKNOWN", "reasons": []})
    if not risk.get("allowed", False):
        signal["signal"] = "HOLD"
        signal["regime"] = "INSUFFICIENT_DATA"
        signal["confidence"] = 0.0
        signal["score"] = 0.0
        signal["reasons"] = [*signal.get("reasons", []), risk.get("reason") or "RISK_REJECTION"]

    payload = {
        "asset": signal.get("asset") or "BTC",
        "signal": signal.get("signal") or "HOLD",
        "signal_id": signal.get("signal_id") or "btc-signal",
        "timestamp": signal.get("signal_timestamp") or snapshot.get("generated_at", datetime.now(timezone.utc)).isoformat(),
        "expires_at": signal.get("signal_expiry") or (datetime.now(timezone.utc)).isoformat(),
        "confidence": float(signal.get("confidence", 0.0) or 0.0),
        "score": float(signal.get("score", 0.0) or 0.0),
        "regime": signal.get("regime") or "UNKNOWN",
        "entry_reference": signal.get("entry_reference") or _btc_reference_price(contracts),
        "features": signal.get("features", snapshot.get("features", {})),
        "risk": risk,
        "data_quality": signal.get("data_quality", snapshot.get("data_quality", {"sufficient": False, "missing_fields": []})),
        "execution": {"paper_trading": True, "live_trading": False, "mode": "PAPER"},
        "reasons": signal.get("reasons", []),
    }
    payload["execution"]["action"] = payload["signal"]
    payload["execution"]["asset"] = payload["asset"]
    return payload


def build_gold_signal_response() -> dict[str, Any]:
    return {
        "asset": "XAUUSD",
        "signal": "HOLD",
        "regime": "INSUFFICIENT_DATA",
        "score": 0.0,
        "confidence": 0.0,
        "signal_id": "gold-disabled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc)).isoformat(),
        "entry_reference": None,
        "features": {},
        "risk": {"allowed": False, "reason": "DEPENDENCY_UNAVAILABLE", "reasons": ["Gold options provider is unavailable because TVdataOptionOI is not installed in the project environment."]},
        "data_quality": {
            "sufficient": False,
            "reason": "GOLD_OPTIONS_PROVIDER_UNAVAILABLE",
            "missing_fields": ["tvdataoption"],
        },
        "execution": {"paper_trading": True, "live_trading": False, "mode": "PAPER"},
        "reasons": ["Gold option signal is disabled; TVdataOptionOI is unavailable and no substitute provider is configured."],
    }


def build_signal_status_response() -> dict[str, Any]:
    try:
        btc_contracts = options_service.fetch_btc_options("BTC")
        btc_available = bool(btc_contracts)
        btc_last_update = max((contract.timestamp for contract in btc_contracts if getattr(contract, "timestamp", None) is not None), default=None)
    except Exception:
        btc_contracts = []
        btc_available = False
        btc_last_update = None
    try:
        GoldOptionsConnector()
        gold_available = False
        gold_reason = "DEPENDENCY_UNAVAILABLE"
    except Exception:
        gold_available = False
        gold_reason = "DEPENDENCY_UNAVAILABLE"
    return {
        "btc": {
            "provider": "deribit",
            "available": btc_available,
            "last_update": btc_last_update.isoformat() if btc_last_update else None,
            "contracts": len(btc_contracts),
        },
        "gold": {
            "provider": "TVdataOptionOI",
            "available": gold_available,
            "reason": gold_reason,
        },
        "signal_engine": {"available": True},
        "execution": {"paper_trading": True, "live_trading": False},
    }


@app.get("/api/signals/btc")
def btc_signal_endpoint():
    return build_btc_signal_response()


@app.get("/api/signals/gold")
def gold_signal_endpoint():
    return build_gold_signal_response()


@app.get("/api/signals/status")
def signal_status_endpoint():
    return build_signal_status_response()


def _safe(value: Any):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
