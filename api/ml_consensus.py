import os
from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from core.data_processor import MarketDataProcessor
from core.trading_engine import TradingEngine
from ml.predictor import ModelPredictor

load_dotenv()

ml_router = APIRouter()
processor = MarketDataProcessor()
prediction_threshold = float(os.getenv("MODEL_THRESHOLD", "0.62"))
h1_engine = TradingEngine(ModelPredictor(
    os.getenv("NEXA_H1_MODEL_PATH", "models/xauusd_h1_pivot_reversal_20261007.joblib"),
    prediction_threshold,
))
m15_engine = TradingEngine(ModelPredictor(
    os.getenv("NEXA_M15_MODEL_PATH", "models/xauusd_m15_pivot_reversal_20261007.joblib"),
    prediction_threshold,
    os.getenv("NEXA_M15_INTERVAL_MODEL_PATH", ""),
))
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MultiTimeframeCandleBatch(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    h1_candles: list[Candle] = Field(min_length=260)
    m15_candles: list[Candle] = Field(min_length=260)


def _prepare_and_analyze(candles: list[Candle], selected_engine: TradingEngine, use_structure_filters=True):
    prepared = processor.prepare([
        {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ])
    return selected_engine.analyze(prepared, use_structure_filters=use_structure_filters)


def verify_api_key(
    x_api_key: str | None,
    allow_loopback=False,
    client_host=None,
    api_key: str | None = None,
):
    configured_api_key = INGEST_API_KEY if api_key is None else api_key
    if configured_api_key and x_api_key != configured_api_key:
        allow_local_prediction = (
            allow_loopback
            and os.getenv("ALLOW_LOCAL_ML_PREDICTIONS", "false").lower() == "true"
            and client_host in {"127.0.0.1", "::1", "localhost", "testclient"}
        )
        if not allow_local_prediction:
            raise HTTPException(status_code=401, detail="Invalid API key.")


@ml_router.post("/api/ml/predict/consensus")
def ml_predict_consensus_endpoint(
    batch: MultiTimeframeCandleBatch,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(
        x_api_key,
        allow_loopback=True,
        client_host=request.client.host if request.client else None,
    )
    try:
        h1 = _prepare_and_analyze(batch.h1_candles, h1_engine, use_structure_filters=False)
        m15 = _prepare_and_analyze(batch.m15_candles, m15_engine, use_structure_filters=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h1_signal = h1.get("signal", "WAIT")
    m15_signal = m15.get("signal", "WAIT")
    h1_setup = h1.get("setup_type", "unknown")
    m15_setup = m15.get("setup_type", "unknown")
    m15_direction = m15.get("direction", "NONE")
    if m15_direction not in {"BUY", "SELL"}:
        m15_direction = m15_signal if m15_signal in {"BUY", "SELL"} else "NONE"
    swing_ready = (
        h1_signal in {"BUY", "SELL"}
        and h1_signal == m15_signal
        and (
            h1_setup in {"continuation", "breakout"}
            or (h1_setup == "reversal" and m15_setup == "reversal")
        )
    )
    normalized_symbol = batch.symbol.upper()
    is_gold_symbol = "XAUUSD" in normalized_symbol or "GOLD" in normalized_symbol
    scalp_ready = (
        is_gold_symbol
        and h1_setup == "no_edge"
        and m15_direction in {"BUY", "SELL"}
    )
    recommended_mode = "SWING" if swing_ready else "SCALP" if scalp_ready else "WAIT"
    signal = (
        m15_signal if recommended_mode == "SWING"
        else m15_direction if recommended_mode == "SCALP"
        else "WAIT"
    )
    confidence = float(m15.get("confidence", 0.0) or 0.0) if signal != "WAIT" else 0.0
    buy_probability = min(
        float(h1.get("probabilities", {}).get("buy", 0.0) or 0.0),
        float(m15.get("probabilities", {}).get("buy", 0.0) or 0.0),
    )
    sell_probability = min(
        float(h1.get("probabilities", {}).get("sell", 0.0) or 0.0),
        float(m15.get("probabilities", {}).get("sell", 0.0) or 0.0),
    )
    m15_probabilities = m15.get("probabilities", {}) or {}
    m15_levels = m15.get("levels", {}) or {}
    bullish_reversal = m15_levels.get("potential_bullish_reversal") or {}
    bearish_reversal = m15_levels.get("potential_bearish_reversal") or {}
    entry_zone = bullish_reversal if signal == "BUY" else bearish_reversal if signal == "SELL" else {}
    entry_zone_low = float(entry_zone.get("low") or 0.0)
    entry_zone_high = float(entry_zone.get("high") or 0.0)
    entry_zone_mid = float(entry_zone.get("mid") or 0.0)
    entry_candle_low = float(m15.get("candle_low") or 0.0)
    entry_candle_high = float(m15.get("candle_high") or 0.0)
    valid_entry_zone = (
        entry_zone_low > 0
        and entry_zone_high >= entry_zone_low
        and entry_candle_high >= entry_candle_low > 0
    )
    entry_triggered = bool(
        recommended_mode in {"SWING", "SCALP"}
        and signal in {"BUY", "SELL"}
        and valid_entry_zone
        and entry_candle_low <= entry_zone_high
        and entry_candle_high >= entry_zone_low
    )
    entry_status = (
        "ZONE_TOUCHED" if entry_triggered
        else "NO_DIRECTION" if signal not in {"BUY", "SELL"}
        else "WAIT_FOR_ZONE" if valid_entry_zone
        else "NO_ENTRY_ZONE"
    )
    h1_version = h1.get("model_version", "unknown")
    m15_version = m15.get("model_version", "unknown")

    return {
        "symbol": batch.symbol.upper(),
        "timeframe": "H1+M15",
        "signal": signal,
        "signal_reason": "H1_M15_REVERSAL_AGREE" if recommended_mode == "SWING" and h1_setup == "reversal" else "H1_M15_AGREE" if recommended_mode == "SWING" else "H1_NO_EDGE_M15_ENTRY" if recommended_mode == "SCALP" else "H1_M15_DISAGREE_OR_WAIT" if h1_signal in {"BUY", "SELL"} and m15_signal in {"BUY", "SELL"} else "NO_VALID_MODE_SETUP",
        "confidence": round(confidence, 2),
        "setup_type": m15_setup if signal != "WAIT" else "unknown",
        "direction": signal if signal != "WAIT" else "NONE",
        "recommended_mode": recommended_mode,
        "planned_direction": signal,
        "entry_direction": signal if entry_triggered else "WAIT",
        "entry_triggered": int(entry_triggered),
        "entry_status": entry_status,
        "entry_price": entry_zone_mid if valid_entry_zone else 0.0,
        "entry_zone_low": entry_zone_low if valid_entry_zone else 0.0,
        "entry_zone_high": entry_zone_high if valid_entry_zone else 0.0,
        "entry_candle_low": entry_candle_low,
        "entry_candle_high": entry_candle_high,
        "entry_zone_source": (
            m15_levels.get("potential_bullish_reversal_source", "unavailable")
            if signal == "BUY"
            else m15_levels.get("potential_bearish_reversal_source", "unavailable")
            if signal == "SELL"
            else "unavailable"
        ),
        "setup_probability": round(confidence / 100.0, 4),
        "buy": round(buy_probability, 2),
        "sell": round(sell_probability, 2),
        "buy_probability": round(buy_probability / 100.0, 4),
        "sell_probability": round(sell_probability / 100.0, 4),
        "h1_signal": h1_signal,
        "h1_confidence": round(float(h1.get("confidence", 0.0) or 0.0), 2),
        "h1_setup_type": h1_setup,
        "h1_model_version": h1_version,
        "h1_candle_time": h1.get("timestamp", ""),
        "m15_signal": m15_signal,
        "m15_confidence": round(float(m15.get("confidence", 0.0) or 0.0), 2),
        "m15_setup_type": m15_setup,
        "m15_atr": m15.get("atr"),
        "grid_interval_buy_atr": m15.get("grid_interval_buy_atr"),
        "grid_interval_sell_atr": m15.get("grid_interval_sell_atr"),
        "swing_stop_buy_atr": m15.get("swing_stop_buy_atr"),
        "swing_target_buy_atr": m15.get("swing_target_buy_atr"),
        "swing_stop_sell_atr": m15.get("swing_stop_sell_atr"),
        "swing_target_sell_atr": m15.get("swing_target_sell_atr"),
        "m15_model_version": m15_version,
        "m15_candle_time": m15.get("timestamp", ""),
        "model_version": f"H1:{h1_version}|M15:{m15_version}",
        "probability_reversal": m15_probabilities.get("probability_reversal", 0.0),
        "probability_continuation": m15_probabilities.get("probability_continuation", 0.0),
        "probability_breakout": m15_probabilities.get("probability_breakout", 0.0),
        "probability_failed_breakout": m15_probabilities.get("probability_failed_breakout", 0.0),
        "probability_no_edge": m15_probabilities.get("probability_no_edge", 0.0),
        "probability_long": m15_probabilities.get("probability_long", 0.0),
        "probability_short": m15_probabilities.get("probability_short", 0.0),
        "probability_flat": m15_probabilities.get("probability_flat", 0.0),
        "probability_reversal_buy": m15_probabilities.get("probability_reversal_buy", 0.0),
        "probability_reversal_sell": m15_probabilities.get("probability_reversal_sell", 0.0),
        "probability_continuation_buy": m15_probabilities.get("probability_continuation_buy", 0.0),
        "probability_continuation_sell": m15_probabilities.get("probability_continuation_sell", 0.0),
        "probability_breakout_buy": m15_probabilities.get("probability_breakout_buy", 0.0),
        "probability_breakout_sell": m15_probabilities.get("probability_breakout_sell", 0.0),
        "probability_failed_breakout_buy": m15_probabilities.get("probability_failed_breakout_buy", 0.0),
        "probability_failed_breakout_sell": m15_probabilities.get("probability_failed_breakout_sell", 0.0),
        "potential_bullish_reversal_low": bullish_reversal.get("low", 0.0),
        "potential_bullish_reversal_high": bullish_reversal.get("high", 0.0),
        "potential_bearish_reversal_low": bearish_reversal.get("low", 0.0),
        "potential_bearish_reversal_high": bearish_reversal.get("high", 0.0),
        "potential_bullish_reversal_source": m15_levels.get("potential_bullish_reversal_source", "unavailable"),
        "potential_bearish_reversal_source": m15_levels.get("potential_bearish_reversal_source", "unavailable"),
        "breakout_up": m15_levels.get("breakout_up") or 0.0,
        "breakout_down": m15_levels.get("breakout_down") or 0.0,
    }
