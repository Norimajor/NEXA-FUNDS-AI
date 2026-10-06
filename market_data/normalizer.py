from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .models import OptionContract


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coalesce(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def _normalize_option_type(value: Any, fallback_symbol: str | None = None) -> str | None:
    if value is None:
        if fallback_symbol:
            last = str(fallback_symbol).upper().rstrip()
            if last.endswith("C"):
                return "call"
            if last.endswith("P"):
                return "put"
        return None
    normalized = str(value).strip().lower()
    if normalized in {"call", "c", "calls"}:
        return "call"
    if normalized in {"put", "p", "puts"}:
        return "put"
    return normalized


def _safe_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            magnitude = float(value)
            if magnitude > 1_000_000_000_000:
                magnitude /= 1000.0
            return datetime.fromtimestamp(magnitude, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            try:
                return datetime.fromtimestamp(int(text) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                dt = datetime.fromisoformat(candidate)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    return None


def _normalize_expiration(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        dt = _safe_datetime(value)
        if dt is not None:
            return dt.strftime("%Y-%m-%d")
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) == 8 and text.isdigit():
            return text
        if "-" in text or "/" in text:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
            try:
                dt = datetime.strptime(text, "%Y/%m/%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return text
    return None


def _days_to_expiration(expiration: Any, reference_time: datetime | None = None) -> int | None:
    if expiration is None:
        return None
    if isinstance(expiration, (int, float)):
        expiration_dt = _safe_datetime(expiration)
    else:
        expiration_dt = _safe_datetime(expiration)
        if expiration_dt is None:
            if isinstance(expiration, str):
                text = expiration.strip()
                if len(text) == 8 and text.isdigit():
                    try:
                        expiration_dt = datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
                    except ValueError:
                        return None
                elif len(text) == 10 and text.count("-") == 2:
                    try:
                        expiration_dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
                    except ValueError:
                        return None
    if expiration_dt is None:
        return None
    reference = reference_time or datetime.now(timezone.utc)
    delta = expiration_dt - reference
    return max(0, int(delta.days))


def normalize_deribit_contract(raw: Mapping[str, Any]) -> OptionContract:
    instrument_name = _coalesce(raw, "instrument_name", "contract_symbol", "symbol")
    fallback = str(instrument_name) if instrument_name else None
    option_type = _normalize_option_type(_coalesce(raw, "option_type", "type"), fallback)
    strike = _safe_float(_coalesce(raw, "strike", "strike_price"))
    underlying_price = _safe_float(_coalesce(raw, "underlying_price", "index_price", "underlying"))
    expiration_value = _coalesce(raw, "expiration", "expiration_timestamp", "expiration_date")
    expiration = _normalize_expiration(expiration_value)
    timestamp_value = _coalesce(raw, "timestamp", "creation_timestamp", "last_update", "updated_at")
    timestamp = _safe_datetime(timestamp_value) or datetime.now(timezone.utc)
    days_to_expiration = _safe_int(_coalesce(raw, "days_to_expiration", "dte"))
    if days_to_expiration is None:
        days_to_expiration = _days_to_expiration(expiration_value or expiration, timestamp)

    greeks = raw.get("greeks") if isinstance(raw.get("greeks"), Mapping) else {}
    implied_volatility = _safe_float(_coalesce(raw, "implied_volatility", "iv"))
    bid_iv = _safe_float(_coalesce(raw, "bid_iv"))
    ask_iv = _safe_float(_coalesce(raw, "ask_iv"))
    if implied_volatility is None:
        implied_volatility = _safe_float(_coalesce(greeks, "implied_volatility", "iv"))

    contract = OptionContract(
        asset="BTC",
        provider="deribit",
        timestamp=timestamp,
        underlying_price=underlying_price,
        expiration=expiration,
        days_to_expiration=days_to_expiration,
        strike=strike,
        option_type=option_type,
        contract_symbol=str(instrument_name) if instrument_name is not None else None,
        bid=_safe_float(_coalesce(raw, "bid", "bid_price")),
        ask=_safe_float(_coalesce(raw, "ask", "ask_price")),
        last_price=_safe_float(_coalesce(raw, "last_price", "last")),
        mark_price=_safe_float(_coalesce(raw, "mark_price", "mark")),
        volume=_safe_float(_coalesce(raw, "volume", "total_volume")),
        open_interest=_safe_float(_coalesce(raw, "open_interest", "oi")),
        implied_volatility=implied_volatility,
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        delta=_safe_float(_coalesce(greeks, "delta")),
        gamma=_safe_float(_coalesce(greeks, "gamma")),
        theta=_safe_float(_coalesce(greeks, "theta")),
        vega=_safe_float(_coalesce(greeks, "vega")),
        moneyness=_safe_float(_coalesce(raw, "moneyness")),
        iv=implied_volatility,
    )
    return contract


def normalize_gold_contract(raw: Mapping[str, Any]) -> OptionContract:
    symbol = _coalesce(raw, "symbol", "contract_symbol")
    fallback_symbol = str(symbol) if symbol is not None else None
    option_type = _normalize_option_type(_coalesce(raw, "option_type", "type", "optionType"), fallback_symbol)
    if option_type is None and isinstance(symbol, str):
        option_type = _normalize_option_type(symbol[-1:], fallback_symbol)

    expiration_value = _coalesce(raw, "expiration", "expiration_date", "expiry")
    expiration = _normalize_expiration(expiration_value)
    if expiration is None and isinstance(symbol, str):
        match = "".join(ch for ch in symbol if ch.isdigit())
        if len(match) >= 8:
            expiration = match[:8]

    timestamp_value = _coalesce(raw, "timestamp", "datetime", "last_update")
    timestamp = _safe_datetime(timestamp_value) or datetime.now(timezone.utc)
    days_to_expiration = _safe_int(_coalesce(raw, "days_to_expiration", "dte"))
    if days_to_expiration is None:
        days_to_expiration = _days_to_expiration(expiration_value or expiration, timestamp)

    implied_volatility = _safe_float(_coalesce(raw, "implied_volatility", "iv"))
    bid_iv = _safe_float(_coalesce(raw, "bid_iv"))
    ask_iv = _safe_float(_coalesce(raw, "ask_iv"))

    contract = OptionContract(
        asset="XAU",
        provider="tvdataoption",
        timestamp=timestamp,
        underlying_price=_safe_float(_coalesce(raw, "underlying_price", "underlying")),
        expiration=expiration,
        days_to_expiration=days_to_expiration,
        strike=_safe_float(_coalesce(raw, "strike")),
        option_type=option_type,
        contract_symbol=str(symbol) if symbol is not None else None,
        bid=_safe_float(_coalesce(raw, "bid", "bid_price")),
        ask=_safe_float(_coalesce(raw, "ask", "ask_price")),
        last_price=_safe_float(_coalesce(raw, "last_price", "last")),
        mark_price=_safe_float(_coalesce(raw, "mark_price", "mid_price", "theo_price")),
        volume=_safe_float(_coalesce(raw, "volume", "total_volume")),
        open_interest=_safe_float(_coalesce(raw, "open_interest", "oi")),
        implied_volatility=implied_volatility,
        bid_iv=bid_iv,
        ask_iv=ask_iv,
        delta=_safe_float(_coalesce(raw, "delta")),
        gamma=_safe_float(_coalesce(raw, "gamma")),
        theta=_safe_float(_coalesce(raw, "theta")),
        vega=_safe_float(_coalesce(raw, "vega")),
        moneyness=_safe_float(_coalesce(raw, "moneyness")),
        iv=implied_volatility,
    )
    return contract
