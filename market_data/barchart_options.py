from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from market_data.models import OptionContract

logger = logging.getLogger("nexafunds.market_data.barchart")

DEFAULT_GOLD_URL = "https://www.barchart.com/futures/options/GC"


@dataclass
class BarchartRawOption:
    raw_symbol: str | None
    source_url: str
    scraped_at: datetime
    raw: dict[str, Any] = field(default_factory=dict)
    source: str = "barchart_scrape"
    raw_expiration: str | None = None
    raw_option_type: str | None = None
    raw_strike: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "raw_symbol": self.raw_symbol,
            "source": self.source,
            "source_url": self.source_url,
            "scraped_at": self.scraped_at.isoformat(),
            "raw": self.raw,
        }
        if self.raw_expiration is not None:
            payload["raw_expiration"] = self.raw_expiration
        if self.raw_option_type is not None:
            payload["raw_option_type"] = self.raw_option_type
        if self.raw_strike is not None:
            payload["raw_strike"] = self.raw_strike
        return payload


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "null", "none", "--", "-", "nan"}:
        return None
    cleaned = text.replace(",", "").replace("%", "").replace("$", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_strike(value: Any) -> float | None:
    return parse_numeric(value)


def parse_expiration(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "null", "none", "--", "-"}:
        return None
    compact = re.sub(r"[^0-9]", "", text)
    if len(compact) == 8 and compact.isdigit():
        try:
            return datetime.strptime(compact, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    for candidate in candidates:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    return None


def parse_option_type(value: Any, symbol: str | None = None) -> str | None:
    if value is None:
        return parse_call_put(symbol)
    text = str(value).strip().lower()
    if text in {"call", "c", "calls", "call option"}:
        return "call"
    if text in {"put", "p", "puts", "put option"}:
        return "put"
    if text in {"buy", "bullish"}:
        return "call"
    if text in {"sell", "bearish"}:
        return "put"
    if symbol is not None:
        parsed = parse_call_put(symbol)
        if parsed:
            return parsed
    return None


def parse_call_put(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    if not text:
        return None
    if text.endswith("C"):
        return "call"
    if text.endswith("P"):
        return "put"
    if "CALL" in text:
        return "call"
    if "PUT" in text:
        return "put"
    matches = re.findall(r"([CP])(?=\d)", text)
    if matches:
        return "call" if matches[-1] == "C" else "put"
    return None


class BarchartOptionsProvider:
    def __init__(
        self,
        symbol: str = "GC",
        asset: str = "GOLD",
        proxy_for: str = "XAUUSD",
        source_url: str = DEFAULT_GOLD_URL,
        timeout_seconds: float = 20.0,
        cache_seconds: float = 30.0,
    ) -> None:
        self.symbol = symbol
        self.asset = asset
        self.proxy_for = proxy_for
        self.source_url = source_url
        self.timeout_seconds = timeout_seconds
        self.cache_seconds = cache_seconds
        self._lock = threading.RLock()
        self._cache: dict[str, Any] | None = None
        self._cache_expires_at: datetime | None = None

    def fetch_raw_options(
        self,
        symbol: str | None = None,
        asset: str | None = None,
        proxy_for: str | None = None,
        max_rows: int | None = None,
    ) -> list[BarchartRawOption]:
        snapshot = self.scrape_snapshot(symbol=symbol or self.symbol, asset=asset or self.asset, proxy_for=proxy_for or self.proxy_for, max_rows=max_rows)
        return snapshot.get("raw_records", [])

    def fetch_contracts(
        self,
        symbol: str | None = None,
        asset: str | None = None,
        proxy_for: str | None = None,
        max_rows: int | None = None,
    ) -> list[OptionContract]:
        snapshot = self.scrape_snapshot(symbol=symbol or self.symbol, asset=asset or self.asset, proxy_for=proxy_for or self.proxy_for, max_rows=max_rows)
        return snapshot.get("contracts", [])

    def fetch_options(
        self,
        symbol: str | None = None,
        asset: str | None = None,
        proxy_for: str | None = None,
        max_rows: int | None = None,
    ) -> list[OptionContract]:
        return self.fetch_contracts(symbol=symbol, asset=asset, proxy_for=proxy_for, max_rows=max_rows)

    def scrape_snapshot(
        self,
        symbol: str | None = None,
        asset: str | None = None,
        proxy_for: str | None = None,
        max_rows: int | None = None,
        allow_cache: bool = True,
    ) -> dict[str, Any]:
        target_symbol = symbol or self.symbol
        target_asset = asset or self.asset
        target_proxy = proxy_for or self.proxy_for
        now = datetime.now(timezone.utc)
        with self._lock:
            if allow_cache and self._cache is not None and self._cache_expires_at is not None and now < self._cache_expires_at:
                snapshot = dict(self._cache)
                snapshot["status"] = dict(snapshot.get("status", {}))
                snapshot["status"]["stale"] = False
                snapshot["status"]["cache_age_seconds"] = round(max(0.0, (now - snapshot.get("generated_at", now)).total_seconds()), 3)
                return snapshot

        start = time.perf_counter()
        try:
            html = self._fetch_public_page(self.source_url)
            raw_records = self._raw_records_from_html(html)
            if not raw_records:
                status = self._status_result(
                    success=False,
                    symbol=target_symbol,
                    records=0,
                    fields_available=[],
                    fields_missing=["contract_symbol", "strike", "expiration", "option_type", "bid", "ask", "last_price", "volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "underlying_price"],
                    error="BARCHART_RESTRICTED_DATA",
                    scrape_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
                )
                snapshot = {
                    "success": False,
                    "provider": "barchart_scrape",
                    "asset": target_asset,
                    "proxy_for": target_proxy,
                    "symbol": target_symbol,
                    "generated_at": now,
                    "contracts": [],
                    "raw_records": [],
                    "status": status,
                }
                self._cache = snapshot
                self._cache_expires_at = now + timedelta(seconds=self.cache_seconds)
                return snapshot

            normalized = self._normalize_rows(raw_records, asset=target_asset, proxy_for=target_proxy)
            if max_rows is not None:
                normalized = normalized[:max_rows]
            status = self._status_result(
                success=bool(normalized),
                symbol=target_symbol,
                records=len(normalized),
                fields_available=self._collect_fields(normalized),
                fields_missing=self._missing_fields(normalized),
                error=None if normalized else "NO_OPTION_ROWS_FOUND",
                scrape_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
            )
            snapshot = {
                "success": bool(normalized),
                "provider": "barchart_scrape",
                "asset": target_asset,
                "proxy_for": target_proxy,
                "symbol": target_symbol,
                "generated_at": now,
                "contracts": normalized,
                "raw_records": raw_records,
                "status": status,
            }
            self._cache = snapshot
            self._cache_expires_at = now + timedelta(seconds=self.cache_seconds)
            return snapshot
        except TimeoutError as exc:
            status = self._status_result(
                success=False,
                symbol=target_symbol,
                records=0,
                fields_available=[],
                fields_missing=["contract_symbol", "strike", "expiration", "option_type", "bid", "ask", "last_price", "mark_price", "volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "underlying_price"],
                error="BARCHART_TIMEOUT",
                scrape_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
            )
            status["detail"] = str(exc)
            snapshot = {
                "success": False,
                "provider": "barchart_scrape",
                "asset": target_asset,
                "proxy_for": target_proxy,
                "symbol": target_symbol,
                "generated_at": now,
                "contracts": [],
                "raw_records": [],
                "status": status,
            }
            self._cache = snapshot
            self._cache_expires_at = now + timedelta(seconds=self.cache_seconds)
            return snapshot
        except Exception as exc:  # pragma: no cover - provider guard
            status = self._status_result(
                success=False,
                symbol=target_symbol,
                records=0,
                fields_available=[],
                fields_missing=["contract_symbol", "strike", "expiration", "option_type", "bid", "ask", "last_price", "mark_price", "volume", "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "underlying_price"],
                error="BARCHART_RESTRICTED_DATA",
                scrape_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
            )
            status["detail"] = str(exc)
            snapshot = {
                "success": False,
                "provider": "barchart_scrape",
                "asset": target_asset,
                "proxy_for": target_proxy,
                "symbol": target_symbol,
                "generated_at": now,
                "contracts": [],
                "raw_records": [],
                "status": status,
            }
            self._cache = snapshot
            self._cache_expires_at = now + timedelta(seconds=self.cache_seconds)
            return snapshot

    def get_status(self, symbol: str | None = None, asset: str | None = None, proxy_for: str | None = None) -> dict[str, Any]:
        target_symbol = symbol or self.symbol
        target_asset = asset or self.asset
        target_proxy = proxy_for or self.proxy_for
        with self._lock:
            if self._cache is not None and self._cache_expires_at is not None and datetime.now(timezone.utc) < self._cache_expires_at:
                status = dict(self._cache.get("status", {}))
                status["stale"] = False
                status["cache_age_seconds"] = round(max(0.0, (datetime.now(timezone.utc) - self._cache.get("generated_at", datetime.now(timezone.utc))).total_seconds()), 3)
                return {
                    "success": status.get("success", False),
                    "provider": "barchart_scrape",
                    "symbol": target_symbol,
                    "asset": target_asset,
                    "proxy_for": target_proxy,
                    "records": len(self._cache.get("contracts", [])),
                    "fields_available": status.get("fields_available", []),
                    "fields_missing": status.get("fields_missing", []),
                    "scrape_time_ms": status.get("scrape_time_ms", 0),
                    "error": status.get("error"),
                    "stale": False,
                    "cache_age_seconds": status["cache_age_seconds"],
                    "last_update": self._cache.get("generated_at").isoformat() if isinstance(self._cache.get("generated_at"), datetime) else None,
                }
        snapshot = self.scrape_snapshot(symbol=target_symbol, asset=target_asset, proxy_for=target_proxy, allow_cache=False)
        return snapshot.get("status", {})

    def _fetch_public_page(self, url: str) -> str:
        logger.info("Fetching public Barchart page: %s", url)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                lower = body.lower()
                if any(token in lower for token in ("login", "sign in", "subscribe", "captcha", "premium access", "pro access")):
                    raise RuntimeError("BARCHART_RESTRICTED_DATA")
                return body
        except urllib.error.URLError as exc:
            raise TimeoutError(f"Public Barchart page is unreachable: {exc}") from exc

    def _raw_records_from_html(self, html: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pattern in (
            r"<script[^>]*id=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>",
            r"<script[^>]*>\s*(\{.*?\})\s*</script>",
        ):
            for match in re.finditer(pattern, html, flags=re.DOTALL | re.IGNORECASE):
                chunk = match.group(1)
                try:
                    payload = json.loads(chunk)
                    rows.extend(self._collect_row_like_payloads(payload))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        return rows

    def _collect_row_like_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            rows: list[dict[str, Any]] = []
            for item in payload:
                rows.extend(self._collect_row_like_payloads(item))
            return rows
        if not isinstance(payload, dict):
            return []
        keys = {"symbol", "contractSymbol", "contract_symbol", "optionType", "option_type", "strike", "expiration", "expiry", "bid", "ask", "last", "volume", "openInterest", "open_interest", "impliedVolatility", "implied_volatility", "delta", "gamma", "theta", "vega"}
        if keys.intersection(payload.keys()):
            return [dict(payload)]
        rows: list[dict[str, Any]] = []
        for value in payload.values():
            rows.extend(self._collect_row_like_payloads(value))
        return rows

    def _normalize_rows(self, rows: Iterable[dict[str, Any]], asset: str, proxy_for: str) -> list[OptionContract]:
        contracts: list[OptionContract] = []
        seen: set[str] = set()
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            symbol = raw.get("symbol") or raw.get("contractSymbol") or raw.get("contract_symbol") or raw.get("name")
            strike_value = raw.get("strike") or raw.get("strikePrice") or raw.get("strike_price")
            expiration_value = raw.get("expiration") or raw.get("expiry") or raw.get("expirationDate") or raw.get("date")
            if symbol is None or strike_value is None or expiration_value is None:
                continue
            option_type = parse_option_type(raw.get("optionType") or raw.get("option_type") or raw.get("type") or raw.get("side"), symbol=symbol)
            if option_type is None:
                continue
            strike = parse_strike(strike_value)
            expiration = parse_expiration(expiration_value)
            if strike is None or expiration is None:
                continue
            uid = f"{str(symbol)}:{expiration}:{strike}:{option_type}"
            if uid in seen:
                continue
            seen.add(uid)
            contract = OptionContract(
                asset=asset.upper(),
                provider="barchart_scrape",
                timestamp=datetime.now(timezone.utc),
                underlying_price=parse_numeric(raw.get("underlyingPrice") or raw.get("underlying_price") or raw.get("underlying")),
                expiration=expiration,
                days_to_expiration=self._days_to_expiration(expiration),
                strike=strike,
                option_type=option_type,
                contract_symbol=str(symbol),
                bid=parse_numeric(raw.get("bid") or raw.get("bidPrice") or raw.get("bid_price")),
                ask=parse_numeric(raw.get("ask") or raw.get("askPrice") or raw.get("ask_price")),
                last_price=parse_numeric(raw.get("last") or raw.get("last_price") or raw.get("lastPrice")),
                mark_price=parse_numeric(raw.get("mark") or raw.get("markPrice") or raw.get("mid")),
                volume=parse_numeric(raw.get("volume") or raw.get("totalVolume") or raw.get("volume_oi") or raw.get("openInterest")),
                open_interest=parse_numeric(raw.get("openInterest") or raw.get("open_interest") or raw.get("oi")),
                implied_volatility=parse_numeric(raw.get("impliedVolatility") or raw.get("implied_volatility") or raw.get("iv")),
                bid_iv=parse_numeric(raw.get("bidIv") or raw.get("bid_iv")),
                ask_iv=parse_numeric(raw.get("askIv") or raw.get("ask_iv")),
                delta=parse_numeric(raw.get("delta")),
                gamma=parse_numeric(raw.get("gamma")),
                theta=parse_numeric(raw.get("theta")),
                vega=parse_numeric(raw.get("vega")),
                moneyness=parse_numeric(raw.get("moneyness")),
                iv=parse_numeric(raw.get("impliedVolatility") or raw.get("implied_volatility") or raw.get("iv")),
            )
            contracts.append(contract)
        return contracts

    def _days_to_expiration(self, expiration: str) -> int | None:
        try:
            expiration_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        delta = expiration_dt - datetime.now(timezone.utc)
        return max(0, delta.days)

    def _collect_fields(self, contracts: list[OptionContract]) -> list[str]:
        fields: list[str] = []
        for name in [
            "contract_symbol",
            "strike",
            "expiration",
            "option_type",
            "bid",
            "ask",
            "last_price",
            "mark_price",
            "volume",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "underlying_price",
        ]:
            if any(getattr(contract, name, None) is not None for contract in contracts):
                fields.append(name)
        return fields

    def _missing_fields(self, contracts: list[OptionContract]) -> list[str]:
        all_fields = [
            "contract_symbol",
            "strike",
            "expiration",
            "option_type",
            "bid",
            "ask",
            "last_price",
            "mark_price",
            "volume",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "underlying_price",
        ]
        return [field_name for field_name in all_fields if not any(getattr(contract, field_name, None) is not None for contract in contracts)]

    def _status_result(
        self,
        *,
        success: bool,
        symbol: str,
        records: int,
        fields_available: list[str],
        fields_missing: list[str],
        error: str | None,
        scrape_time_ms: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "success": success,
            "provider": "barchart_scrape",
            "symbol": symbol,
            "records": records,
            "fields_available": fields_available,
            "fields_missing": fields_missing,
            "scrape_time_ms": round(scrape_time_ms, 3),
            "error": error,
            "stale": False,
        }
