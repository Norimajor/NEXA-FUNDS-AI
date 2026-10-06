from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from market_data.options.features import OptionsFeatureEngine
from market_data.options.signal_engine import OptionsSignalEngine
from market_data.options.risk import OptionsRiskManager

from .deribit import DeribitAPIError, DeribitOptionsClient
from .gold_options import GoldOptionsConnector
from .models import OptionContract

logger = logging.getLogger("nexafunds.market_data.service")


class OptionsMarketDataService:
    def __init__(
        self,
        deribit_client: DeribitOptionsClient | None = None,
        gold_connector: GoldOptionsConnector | None = None,
        cache_seconds: float | None = None,
    ) -> None:
        self.deribit_client = deribit_client or DeribitOptionsClient()
        self.gold_connector = gold_connector or GoldOptionsConnector()
        self.cache_seconds = float(cache_seconds if cache_seconds is not None else os.getenv("BTC_OPTIONS_CACHE_SECONDS", "30"))
        self._lock = threading.RLock()
        self._btc_cache: dict[str, Any] | None = None
        self._btc_cache_expires_at: datetime | None = None

    def fetch_btc_options(self, currency: str = "BTC") -> list[OptionContract]:
        logger.info("Fetching BTC options from Deribit for %s", currency)
        return self.deribit_client.fetch_active_options(currency=currency)

    def fetch_gold_options(self, symbol: str = "COMEX:GC1!", series: str | None = None) -> list[OptionContract]:
        logger.info("Fetching gold option surface for %s", symbol)
        return self.gold_connector.fetch_surface(symbol=symbol, series=series)

    def refresh_btc_snapshot(self, currency: str = "BTC") -> dict[str, Any]:
        start = time.perf_counter()
        with self._lock:
            if self._btc_cache is not None and self._btc_cache_expires_at is not None and datetime.now(timezone.utc) < self._btc_cache_expires_at:
                return self._btc_cache

            try:
                contracts = self.fetch_btc_options(currency)
                provider_fetch_ms = (time.perf_counter() - start) * 1000.0
                normalization_start = time.perf_counter()
                normalized = list(contracts)
                normalization_ms = (time.perf_counter() - normalization_start) * 1000.0
                feature_engine = OptionsFeatureEngine()
                feature_start = time.perf_counter()
                feature_vector = feature_engine.build(normalized, asset=currency, reference_price=self._reference_price(normalized))
                feature_ms = (time.perf_counter() - feature_start) * 1000.0
                signal_engine = OptionsSignalEngine(feature_engine=feature_engine)
                signal_start = time.perf_counter()
                signal = signal_engine.generate(normalized, asset=currency, reference_price=self._reference_price(normalized))
                signal_ms = (time.perf_counter() - signal_start) * 1000.0
                risk_start = time.perf_counter()
                risk_manager = OptionsRiskManager({"stale_data_seconds": 300, "signal_max_age_seconds": 600})
                risk = risk_manager.validate_signal(
                    signal,
                    account_equity=100000.0,
                    risk_percentage=0.02,
                    entry_price=self._reference_price(normalized),
                    stop_loss_distance=None,
                    spread_pct=None,
                    signal_age_seconds=0,
                )
                risk_ms = (time.perf_counter() - risk_start) * 1000.0
                snapshot = {
                    "asset": currency,
                    "provider": "deribit",
                    "generated_at": datetime.now(timezone.utc),
                    "contracts": normalized,
                    "underlying_price": self._reference_price(normalized),
                    "features": feature_vector.to_dict(),
                    "signal": signal,
                    "risk": risk,
                    "data_quality": signal.get("data_quality", {"sufficient": False, "missing_fields": []}),
                    "timings_ms": {
                        "provider_fetch_ms": round(provider_fetch_ms, 3),
                        "normalization_ms": round(normalization_ms, 3),
                        "feature_calculation_ms": round(feature_ms, 3),
                        "signal_calculation_ms": round(signal_ms, 3),
                        "risk_calculation_ms": round(risk_ms, 3),
                        "total_request_ms": round((time.perf_counter() - start) * 1000.0, 3),
                    },
                }
                self._btc_cache = snapshot
                self._btc_cache_expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.cache_seconds)
                logger.info("BTC snapshot refreshed; contracts=%s feature_time_ms=%s", len(normalized), snapshot["timings_ms"]["feature_calculation_ms"])
                return snapshot
            except (DeribitAPIError, TimeoutError, RuntimeError) as exc:
                logger.warning("BTC snapshot refresh failed: %s", exc)
                self._btc_cache = None
                self._btc_cache_expires_at = None
                return {
                    "asset": currency,
                    "provider": "deribit",
                    "generated_at": datetime.now(timezone.utc),
                    "contracts": [],
                    "underlying_price": None,
                    "features": {},
                    "signal": {"signal": "HOLD", "regime": "INSUFFICIENT_DATA", "confidence": 0.0, "score": 0.0, "data_quality": {"sufficient": False, "reason": "DERIBIT_TIMEOUT" if isinstance(exc, TimeoutError) else "DERIBIT_ERROR"}},
                    "risk": {"allowed": False, "reason": "DERIBIT_TIMEOUT" if isinstance(exc, TimeoutError) else "DERIBIT_ERROR", "reasons": [str(exc)]},
                    "data_quality": {"sufficient": False, "reason": "DERIBIT_TIMEOUT" if isinstance(exc, TimeoutError) else "DERIBIT_ERROR", "missing_fields": ["deribit_response"]},
                    "timings_ms": {"provider_fetch_ms": 0.0, "normalization_ms": 0.0, "feature_calculation_ms": 0.0, "signal_calculation_ms": 0.0, "risk_calculation_ms": 0.0, "total_request_ms": round((time.perf_counter() - start) * 1000.0, 3)},
                }

    def get_btc_snapshot(self, currency: str = "BTC") -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._btc_cache is not None and self._btc_cache_expires_at is not None and now < self._btc_cache_expires_at:
                return self._btc_cache
        return self.refresh_btc_snapshot(currency=currency)

    def fetch_all(self, btc_currency: str = "BTC", gold_symbol: str = "COMEX:GC1!", gold_series: str | None = None) -> list[OptionContract]:
        contracts: list[OptionContract] = []
        try:
            contracts.extend(self.fetch_btc_options(btc_currency))
        except Exception as exc:  # pragma: no cover - integration guard
            logger.warning("BTC options fetch failed: %s", exc)
        try:
            contracts.extend(self.fetch_gold_options(gold_symbol, gold_series))
        except Exception as exc:  # pragma: no cover - integration guard
            logger.warning("Gold options fetch failed: %s", exc)
        return contracts

    def _reference_price(self, contracts: list[OptionContract]) -> float | None:
        values = [float(contract.underlying_price) for contract in contracts if contract.underlying_price is not None]
        return max(values) if values else None
