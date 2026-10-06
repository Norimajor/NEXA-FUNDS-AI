from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, parse, request

from .normalizer import normalize_deribit_contract

logger = logging.getLogger("nexafunds.market_data.deribit")


class DeribitAPIError(RuntimeError):
    pass


class DeribitOptionsClient:
    def __init__(
        self,
        base_url: str = "https://www.deribit.com/api/v2",
        timeout: float | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.75,
    ) -> None:
        import os

        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout if timeout is not None else os.getenv("DERIBIT_TIMEOUT_SECONDS", "10"))
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch_contracts(self, currency: str = "BTC") -> list:
        return self.fetch_active_options(currency=currency)

    def fetch_active_options(self, currency: str = "BTC", max_contracts: int | None = None) -> list:
        instruments = self.fetch_instruments(currency=currency, kind="option", expired=False)
        if max_contracts is not None and max_contracts > 0 and len(instruments) > max_contracts:
            instruments = self._select_representative_instruments(instruments, max_contracts=max_contracts)
        contracts = []
        for instrument in instruments:
            name = instrument.get("instrument_name") or instrument.get("name")
            if not name:
                continue
            try:
                ticker = self.fetch_ticker(instrument_name=name)
                combined = {**instrument, **ticker}
                contracts.append(normalize_deribit_contract(combined))
            except DeribitAPIError as exc:
                logger.warning("Skipping Deribit contract %s: %s", name, exc)
        return contracts

    def _select_representative_instruments(self, instruments: list[dict[str, Any]], max_contracts: int) -> list[dict[str, Any]]:
        filtered = [instrument for instrument in instruments if instrument.get("strike") is not None and (instrument.get("option_type") or instrument.get("instrument_name") or "")]
        if not filtered:
            return instruments[:max_contracts]
        strikes = [float(item["strike"]) for item in filtered if item.get("strike") is not None]
        if strikes:
            midpoint = sorted(strikes)[len(strikes) // 2]
        else:
            midpoint = 0.0

        ranked = sorted(
            filtered,
            key=lambda item: (
                abs(float(item.get("strike", 0.0)) - midpoint),
                abs(float(item.get("expiration", 0) or 0)),
                str(item.get("instrument_name", "")),
            ),
        )
        return ranked[: max_contracts]

    def fetch_instruments(self, currency: str = "BTC", kind: str = "option", expired: bool = False) -> list[dict[str, Any]]:
        payload = {"currency": currency, "kind": kind, "expired": str(expired).lower()}
        response = self._request("public/get_instruments", payload)
        result = response.get("result")
        if not isinstance(result, list):
            raise DeribitAPIError("Malformed Deribit instrument payload.")
        return result

    def fetch_ticker(self, instrument_name: str) -> dict[str, Any]:
        response = self._request("public/ticker", {"instrument_name": instrument_name})
        result = response.get("result")
        if not isinstance(result, dict):
            raise DeribitAPIError(f"Malformed Deribit ticker payload for {instrument_name!r}.")
        return result

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        if params:
            encoded = parse.urlencode({key: str(value).lower() if isinstance(value, bool) else value for key, value in params.items()})
            url = f"{url}?{encoded}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = request.Request(url, headers={"Accept": "application/json", "User-Agent": "NexaFunds-Options/1.0"})
                with request.urlopen(req, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                    payload = json.loads(body)
                    if response.status == 429:
                        raise DeribitAPIError("Rate limited by Deribit.")
                    if not isinstance(payload, dict):
                        raise DeribitAPIError("Malformed JSON response from Deribit.")
                    if payload.get("success") is False:
                        error_message = payload.get("error", {}).get("message") or "Deribit request failed."
                        raise DeribitAPIError(error_message)
                    return payload
            except (error.HTTPError, error.URLError, json.JSONDecodeError, TimeoutError, DeribitAPIError) as exc:
                last_error = exc
                if isinstance(exc, (TimeoutError, OSError)):
                    logger.warning("Deribit request timed out on %s (attempt %s/%s): %s", endpoint, attempt, self.max_retries, exc)
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_seconds * attempt)
                        continue
                    raise TimeoutError(f"Deribit request timed out for {endpoint} after {self.timeout} seconds") from exc
                if isinstance(exc, error.HTTPError):
                    status = exc.code
                    if status == 429:
                        logger.warning("Deribit rate limit hit on %s (attempt %s/%s).", endpoint, attempt, self.max_retries)
                    elif 500 <= status < 600:
                        logger.warning("Deribit server error %s on %s (attempt %s/%s).", status, endpoint, attempt, self.max_retries)
                    else:
                        logger.warning("Deribit HTTP error %s on %s: %s", status, endpoint, exc)
                elif isinstance(exc, DeribitAPIError):
                    logger.warning("Deribit API error on %s: %s", endpoint, exc)
                else:
                    logger.warning("Connection issue for %s: %s", endpoint, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
                    continue
                break

        raise DeribitAPIError(f"Deribit request failed for {endpoint}: {last_error}")
