from __future__ import annotations

import logging
from typing import Any

from .normalizer import normalize_gold_contract

logger = logging.getLogger("nexafunds.market_data.gold")


class GoldOptionsConnector:
    def __init__(self, symbol: str = "COMEX:GC1!", timeout: int = 20):
        self.symbol = symbol
        self.timeout = timeout
        self._client = None

    def fetch_contracts(self, symbol: str | None = None, series: str | None = None):
        return self.fetch_surface(symbol=symbol, series=series)

    def _ensure_client(self):
        if self._client is None:
            try:
                from tvdataoption import TvDataOption
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "TVdataOptionOI is not installed. Install the library from the project repo or `pip install tvdataoption`."
                ) from exc
            self._client = TvDataOption(anonymous=True, timeout=self.timeout)
        return self._client

    def fetch_surface(self, symbol: str | None = None, series: str | None = None) -> list:
        target_symbol = symbol or self.symbol
        logger.info("Fetching gold option surface for %s", target_symbol)
        try:
            client = self._ensure_client()
            if series:
                frame = client.get_complete_option_surface(symbol=target_symbol, series=series)
            else:
                frame = client.get_complete_option_surface(symbol=target_symbol)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - integration path
            raise RuntimeError(f"Failed to fetch TVdataOptionOI surface for {target_symbol}: {exc}") from exc

        if frame is None:
            return []

        contracts = []
        for _, row in frame.iterrows():
            if row is None:
                continue
            contracts.append(normalize_gold_contract(row.to_dict()))
        return contracts

    def fetch_snapshot(self, symbol: str | None = None) -> list:
        return self.fetch_surface(symbol=symbol)
