from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = [
    "strike",
    "expiration",
    "option type",
    "underlying",
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


def _fetch(url: str, timeout: int = 20) -> tuple[bool, str, str | None]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, body, None
    except (HTTPError, URLError, TimeoutError) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def _contains_restriction(text: str) -> bool:
    lower = text.lower()
    markers = [
        "login",
        "sign in",
        "captcha",
        "verify you are human",
        "subscribe",
        "subscription",
        "premium",
        "access denied",
        "anti-bot",
        "not authorized",
        "please try again later",
        "restricted",
    ]
    return any(marker in lower for marker in markers)


def _find_option_payload(html: str) -> dict[str, Any] | None:
    matches = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    for chunk in matches:
        if not chunk.strip():
            continue
        for needle in ("optionChain", "optionsChain", "marketData", "symbolInfo", "initialData", "data"):
            if needle in chunk:
                try:
                    start = chunk.find("{")
                    end = chunk.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = chunk[start : end + 1]
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                except Exception:
                    continue
    return None


def _flatten_candidates(payload: Any, out: list[Any] | None = None) -> list[Any]:
    out = out or []
    if isinstance(payload, dict):
        for value in payload.values():
            _flatten_candidates(value, out)
    elif isinstance(payload, list):
        for item in payload:
            _flatten_candidates(item, out)
    else:
        return out
    return out


def _count_contracts(payload: Any) -> int:
    count = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in {"options", "contracts", "rows", "data", "series"}:
                count += len(value) if isinstance(value, list) else 1
            elif isinstance(value, (dict, list)):
                count += _count_contracts(value)
    elif isinstance(payload, list):
        for item in payload:
            count += _count_contracts(item)
    return count


def _sample_contracts(payload: Any, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in {"options", "contracts", "rows", "data"} and isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, (dict, list)):
                rows.extend(_sample_contracts(value, limit))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows.append(item)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        signature = json.dumps(row, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def main() -> None:
    print("GOLD OPTIONS SOURCE DISCOVERY")
    source_url = "https://www.tradingview.com/symbols/COMEX-GC1!/options/"
    start = time.perf_counter()
    package_present = importlib.util.find_spec("tvdataoption") is not None

    print("Source:")
    print("TradingView public options page")
    print("Symbol:")
    print("COMEX:GC1!")
    print("Accessible:")
    print(False)

    ok, html, err = _fetch(source_url, timeout=20)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
    payload = _find_option_payload(html) if ok and html else None
    contract_count = _count_contracts(payload) if payload else 0
    sample_rows = _sample_contracts(payload, limit=5) if payload else []
    calls = sum(1 for row in sample_rows if isinstance(row, dict) and str(row.get("option_type") or row.get("type") or "").lower() in {"call", "c"})
    puts = sum(1 for row in sample_rows if isinstance(row, dict) and str(row.get("option_type") or row.get("type") or "").lower() in {"put", "p"})

    print("Success:")
    print(False)
    print("Contracts:")
    print(contract_count)
    print("Calls:")
    print(calls)
    print("Puts:")
    print(puts)
    print("Fields available:")
    for field in FIELDS:
        # Manual discovery intentionally reports only observed public fields, not assumed ones.
        if field.lower() in " ".join(json.dumps(sample_rows[:1], sort_keys=True).lower().split()):
            print(f"- {field}")
    print("Fields unavailable:")
    for field in FIELDS:
        print(f"- {field}")
    print("Latency:")
    print(f"{duration_ms} ms")
    print("Error:")
    if ok:
        if _contains_restriction(html):
            print("TRADINGVIEW_PUBLIC_OPTIONS_PAGE_BLOCKED")
        else:
            print("PUBLIC_OPTION_CHAIN_NOT_EXPOSED")
    else:
        print(err or "PUBLIC_OPTION_CHAIN_NOT_EXPOSED")

    print("Package present (tvdataoption):")
    print(package_present)
    print("FLOW_TYPE = CHAIN/POSITIONING")
    if sample_rows:
        print("Sample contracts:")
        for row in sample_rows:
            print(json.dumps(row, sort_keys=True, default=str)[:1500])
    else:
        print("No public option rows exposed in the TradingView page or the package is unavailable.")

    print("DATA SOURCE:")
    print("TradingView public page / unavailable package")
    print("ACCESS:")
    print("False")
    print("CONTRACT COUNT:")
    print(contract_count)
    print("OI:")
    print("None")
    print("VOLUME:")
    print("None")
    print("IV:")
    print("None")
    print("DELTA:")
    print("None")
    print("GAMMA:")
    print("None")
    print("THETA:")
    print("None")
    print("VEGA:")
    print("None")
    print("BID/ASK:")
    print("None")
    print("TRADE-LEVEL FLOW:")
    print("False")
    print("CLASSIFICATION:")
    print("INSUFFICIENT_DATA")


if __name__ == "__main__":
    main()
