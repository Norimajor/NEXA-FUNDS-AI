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
    trend_path=os.getenv("NEXA_H1_TREND_MODEL_PATH", "models/xauusd_h1_trend_20261008.joblib"),
))
m15_engine = TradingEngine(ModelPredictor(
    os.getenv("NEXA_M15_MODEL_PATH", "models/xauusd_m15_pivot_reversal_20261007.joblib"),
    prediction_threshold,
    os.getenv("NEXA_M15_INTERVAL_MODEL_PATH", ""),
    os.getenv("NEXA_M15_TREND_MODEL_PATH", "models/xauusd_m15_trend_20261008.joblib"),
))
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")


def _setup_context(features: dict, direction: str) -> str:
    aligned = []
    opposed = []
    directional_features = (
        ("ema_distance_20_50", "20/50 EMA alignment"),
        ("ema_distance_50_200", "50/200 EMA alignment"),
        ("return_20", "20-bar momentum"),
        ("swing_high_change_atr", "swing-high progression"),
        ("swing_low_change_atr", "swing-low progression"),
    )
    sign = 1.0 if direction == "BUY" else -1.0
    for name, label in directional_features:
        value = float(features.get(name, 0.0) or 0.0)
        if abs(value) < 1e-8:
            continue
        (aligned if value * sign > 0 else opposed).append(f"{label} {'+' if value > 0 else '-'}")

    structure_key = "structure_bull" if direction == "BUY" else "structure_bear"
    opposite_structure_key = "structure_bear" if direction == "BUY" else "structure_bull"
    if float(features.get(structure_key, 0.0) or 0.0) > 0:
        aligned.append("swing structure agrees")
    elif float(features.get(opposite_structure_key, 0.0) or 0.0) > 0:
        opposed.append("swing structure conflicts")
    sweep_key = "sweep_low" if direction == "BUY" else "sweep_high"
    rejection_key = "bullish_rejection" if direction == "BUY" else "bearish_rejection"
    if float(features.get(sweep_key, 0.0) or 0.0) > 0:
        aligned.append("liquidity sweep")
    if float(features.get(rejection_key, 0.0) or 0.0) > 0:
        aligned.append("directional candle rejection")

    parts = []
    if aligned:
        parts.append("aligned context: " + ", ".join(aligned[:4]))
    if opposed:
        parts.append("conflicting context: " + ", ".join(opposed[:2]))
    return "; ".join(parts) if parts else "no listed context cue; classifier output is the decision basis"


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
    h1_direction = h1.get("trend_direction")
    if h1_direction not in {"BUY", "SELL", "RANGE"}:
        h1_has_learned_setup = h1_setup in {
            "continuation",
            "breakout",
            "reversal",
            "failed_breakout",
        }
        h1_direction = h1.get("direction", h1_signal)
        if (
            h1_direction not in {"BUY", "SELL"}
            or h1_signal != h1_direction
            or not h1_has_learned_setup
        ):
            h1_direction = "RANGE"
    m15_direction = m15.get("direction", m15_signal)
    if m15_direction not in {"BUY", "SELL"} or m15_signal != m15_direction:
        m15_direction = "NONE"
    m15_has_learned_setup = m15_setup in {
        "continuation",
        "breakout",
        "reversal",
        "failed_breakout",
    }
    direction_confirmed = (
        h1_direction in {"BUY", "SELL"}
        and m15_direction == h1_direction
        and m15_has_learned_setup
    )
    reversal_confirmed = (
        h1_direction in {"BUY", "SELL"}
        and m15_direction in {"BUY", "SELL"}
        and m15_direction != h1_direction
        and m15_setup in {"reversal", "failed_breakout"}
    )
    setup_confirmed = direction_confirmed or reversal_confirmed
    recommended_mode = "SWING" if setup_confirmed else "WAIT"
    signal = m15_direction if setup_confirmed else "WAIT"
    if setup_confirmed:
        m15_confidence = float(m15.get("confidence", 0.0) or 0.0)
        confidence = (
            min(float(h1.get("confidence", 0.0) or 0.0), m15_confidence)
            if h1_signal == h1_direction
            else m15_confidence
        )
    else:
        confidence = 0.0
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
    poi_direction = signal if signal in {"BUY", "SELL"} else h1_direction
    entry_zone = bullish_reversal if poi_direction == "BUY" else bearish_reversal if poi_direction == "SELL" else {}
    entry_zone_low = float(entry_zone.get("low") or 0.0)
    entry_zone_high = float(entry_zone.get("high") or 0.0)
    entry_zone_mid = float(entry_zone.get("mid") or 0.0)
    poi_source = (
        m15_levels.get("potential_bullish_reversal_source", "unavailable")
        if poi_direction == "BUY"
        else m15_levels.get("potential_bearish_reversal_source", "unavailable")
        if poi_direction == "SELL"
        else "unavailable"
    )
    entry_candle_low = float(m15.get("candle_low") or 0.0)
    entry_candle_high = float(m15.get("candle_high") or 0.0)
    valid_entry_zone = (
        entry_zone_low > 0
        and entry_zone_high >= entry_zone_low
        and entry_candle_high >= entry_candle_low > 0
    )
    entry_triggered = bool(
        recommended_mode == "SWING"
        and signal in {"BUY", "SELL"}
        and valid_entry_zone
        and entry_candle_low <= entry_zone_high
        and entry_candle_high >= entry_zone_low
    )
    entry_status = (
        "ZONE_TOUCHED" if entry_triggered
        else "WAIT_FOR_M15_CONFIRMATION" if h1_direction in {"BUY", "SELL"} and not setup_confirmed
        else "RANGE_WAIT_FOR_DIRECTION" if h1_direction == "RANGE"
        else "WAIT_FOR_ZONE" if valid_entry_zone
        else "NO_ENTRY_ZONE"
    )
    h1_version = h1.get("model_version", "unknown")
    m15_version = m15.get("model_version", "unknown")
    confirmation_status = (
        "REVERSAL_CONFIRMED" if reversal_confirmed
        else "DIRECTION_CONFIRMED" if direction_confirmed
        else "WAIT_M15_CONFIRMATION" if h1_direction in {"BUY", "SELL"}
        else "RANGE_NO_DIRECTIONAL_BIAS"
    )
    model_class = m15.get(
        "model_class",
        f"{m15_setup}_{m15_direction.lower()}" if m15_direction in {"BUY", "SELL"} else m15_setup,
    )
    model_class_probability = float(
        m15.get(
            "model_class_probability",
            m15.get("setup_probability", float(m15.get("confidence", 0.0) or 0.0) / 100.0),
        )
        or 0.0
    )
    runner_up_class = m15.get("runner_up_class", "unavailable")
    runner_up_probability = float(m15.get("runner_up_probability", 0.0) or 0.0)
    setup_reason = (
        f"M15 top class {model_class} score {model_class_probability:.3f}; "
        f"runner-up {runner_up_class} score {runner_up_probability:.3f}; "
        f"raw signal {m15_signal} ({m15.get('signal_reason', 'not supplied')})"
    )
    context_direction = m15_direction
    if context_direction not in {"BUY", "SELL"}:
        candidate_direction = m15.get("direction", "NONE")
        context_direction = candidate_direction if candidate_direction in {"BUY", "SELL"} else "NONE"
    setup_context = _setup_context(m15.get("features", {}) or {}, context_direction)
    poi_reason = (
        f"{poi_direction} structural POI from {poi_source} at "
        f"{entry_zone_low:g}-{entry_zone_high:g}"
        if poi_direction in {"BUY", "SELL"} and entry_zone_low > 0
        else "No directional structural POI available"
    )
    decision_explanation = (
        f"H1 trend {h1_direction}: {h1.get('trend_reason', 'trend evidence unavailable')}; "
        f"{setup_reason}; context {setup_context}; POI {poi_reason}; entry {entry_status}"
    )

    return {
        "symbol": batch.symbol.upper(),
        "timeframe": "H1+M15",
        "signal": signal,
        "signal_reason": (
            "M15_REVERSAL_CONFIRMED"
            if reversal_confirmed
            else "H1_M15_DIRECTION_CONFIRMED"
            if direction_confirmed
            else "WAIT_FOR_M15_CONFIRMATION"
            if h1_direction in {"BUY", "SELL"}
            else "WAIT_FOR_MARKET_DIRECTION"
        ),
        "confidence": round(confidence, 2),
        "setup_type": m15_setup if signal != "WAIT" else "unknown",
        "direction": signal if signal != "WAIT" else "NONE",
        "trend_direction": h1_direction,
        "trend_strength": round(float(h1.get("trend_strength", 0.0) or 0.0), 3),
        "trend_reason": h1.get("trend_reason", "MODEL_OR_LEGACY_DIRECTION"),
        "trend_model_version": h1.get("trend_model_version", "unavailable"),
        "model_class": model_class,
        "model_class_probability": round(model_class_probability, 4),
        "runner_up_class": runner_up_class,
        "runner_up_probability": round(runner_up_probability, 4),
        "setup_reason": setup_reason,
        "setup_context": setup_context,
        "decision_explanation": decision_explanation,
        "confirmation_status": confirmation_status,
        "recommended_mode": recommended_mode,
        "planned_direction": signal,
        "poi_direction": poi_direction,
        "poi_reason": poi_reason,
        "entry_direction": signal if entry_triggered else "WAIT",
        "entry_triggered": int(entry_triggered),
        "entry_status": entry_status,
        "entry_price": entry_zone_mid if valid_entry_zone else 0.0,
        "entry_zone_low": entry_zone_low if valid_entry_zone else 0.0,
        "entry_zone_high": entry_zone_high if valid_entry_zone else 0.0,
        "entry_candle_low": entry_candle_low,
        "entry_candle_high": entry_candle_high,
        "entry_zone_source": poi_source,
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
