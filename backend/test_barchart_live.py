from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.barchart_options import BarchartOptionsProvider

BLOCKER_TOKENS = (
    "login",
    "sign in",
    "subscribe",
    "subscription",
    "premium access",
    "pro access",
    "captcha",
    "access denied",
    "anti-bot",
    "verify you are human",
    "not authorized",
    "please try again later",
)

FIELD_SET = [
    "strike",
    "expiration",
    "call/put",
    "bid",
    "ask",
    "last",
    "volume",
    "open interest",
    "IV",
    "delta",
    "gamma",
    "theta",
    "vega",
]


def _describe_contract(contract: Any) -> dict[str, Any]:
    if contract is None:
        return {}
    data = contract.to_dict() if hasattr(contract, "to_dict") else dict(contract)
    return {
        "asset": data.get("asset"),
        "provider": data.get("provider"),
        "contract_symbol": data.get("contract_symbol"),
        "strike": data.get("strike"),
        "expiration": data.get("expiration"),
        "option_type": data.get("option_type"),
        "bid": data.get("bid"),
        "ask": data.get("ask"),
        "last_price": data.get("last_price"),
        "volume": data.get("volume"),
        "open_interest": data.get("open_interest"),
        "implied_volatility": data.get("implied_volatility"),
        "delta": data.get("delta"),
        "gamma": data.get("gamma"),
        "theta": data.get("theta"),
        "vega": data.get("vega"),
        "underlying_price": data.get("underlying_price"),
    }


def _print_contracts(contracts: list[Any], limit: int = 5) -> None:
    for index, contract in enumerate(contracts[:limit], start=1):
        print(f"  Contract {index}: {_describe_contract(contract)}")


def _print_fields(fields_available: list[str], fields_unavailable: list[str]) -> None:
    print("Fields available:")
    for field in FIELD_SET:
        if field in fields_available:
            print(f"- {field}")
    print("Fields unavailable:")
    if not fields_unavailable:
        print("- none")
    else:
        for field in fields_unavailable:
            print(f"- {field}")


def main() -> None:
    provider = BarchartOptionsProvider(timeout_seconds=20.0)
    start = time.perf_counter()
    print("BARCHART LIVE TEST")
    print("Provider:")
    print(provider.__class__.__name__)
    print("Symbol:")
    print(provider.symbol)
    print("URL:")
    print(provider.source_url)

    try:
        html = provider._fetch_public_page(provider.source_url)
        raw_rows = provider._raw_records_from_html(html)
        snapshot = provider.scrape_snapshot(symbol=provider.symbol, asset=provider.asset, proxy_for=provider.proxy_for, allow_cache=False)
        status = snapshot.get("status", {})
        error = status.get("error")
        error_code = status.get("error") or "NONE"
        success = bool(snapshot.get("success")) and bool(snapshot.get("contracts"))
        fields_available = list(status.get("fields_available", []))
        fields_missing = list(status.get("fields_missing", []))
        normalized = snapshot.get("contracts", [])
        duration = round((time.perf_counter() - start) * 1000.0, 3)

        print("Success:")
        print(success)
        print("Error:")
        print(error or "NONE")
        print("Error code:")
        print(error_code)
        print("Raw rows:")
        print(len(raw_rows))
        print("Normalized contracts:")
        print(len(normalized))
        print("Fields available:")
        for field in FIELD_SET:
            if field in fields_available:
                print(f"- {field}")
        print("Fields unavailable:")
        if not fields_missing:
            print("- none")
        else:
            for field in fields_missing:
                print(f"- {field}")
        print("Scrape duration:")
        print(f"{duration} ms")

        if success:
            print("Sample normalized contracts:")
            _print_contracts(normalized, limit=5)
        else:
            blocked = any(token in html.lower() for token in BLOCKER_TOKENS)
            if blocked or error in {"BARCHART_RESTRICTED_DATA", "BARCHART_TIMEOUT"}:
                print("BARCHART_RESTRICTED_DATA")
                print("The public Barchart page was blocked or required authentication, paywall, CAPTCHA, or anti-bot handling. No bypass was attempted.")
            else:
                print("HTML loaded but no usable option rows were found. The page was inspected for server-rendered JSON, dynamic data payloads, and blocking markers; no public option dataset was exposed in this environment.")
    except Exception as exc:  # pragma: no cover - manual live test
        duration = round((time.perf_counter() - start) * 1000.0, 3)
        print("Success:")
        print(False)
        print("Error:")
        print(str(exc))
        print("Error code:")
        print("BARCHART_RESTRICTED_DATA")
        print("Raw rows:")
        print(0)
        print("Normalized contracts:")
        print(0)
        print("Fields available:")
        print("- none")
        print("Fields unavailable:")
        for field in FIELD_SET:
            print(f"- {field}")
        print("Scrape duration:")
        print(f"{duration} ms")
        print("BARCHART_RESTRICTED_DATA")
        print("The public Barchart page was blocked, timed out, or returned a restricted/anti-bot response. No bypass was used.")
        raise


if __name__ == "__main__":
    main()
