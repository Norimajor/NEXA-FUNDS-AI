from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def compute_volume_profile(df: pd.DataFrame, value_area_percentage: float = 0.7, bins: int | None = None) -> dict[str, Any]:
    if df.empty:
        return {
            "poc": None,
            "vah": None,
            "val": None,
            "high_volume_nodes": [],
            "low_volume_nodes": [],
            "total_volume": 0.0,
            "value_area_volume_percentage": 0.0,
            "value_area_percentage": value_area_percentage,
            "volume_quality": "INSUFFICIENT_DATA",
            "bin_count": 0,
        }

    hist = df.copy()
    hist = hist.sort_values("timestamp").reset_index(drop=True)
    low = float(hist["low"].min())
    high = float(hist["high"].max())
    if np.isclose(low, high):
        low -= 0.5
        high += 0.5

    bin_count = bins or max(20, min(100, int((high - low) * 10)))
    edges = np.linspace(low, high, bin_count + 1)
    bin_index = np.digitize(hist["close"], edges) - 1
    bin_index = np.clip(bin_index, 0, bin_count - 1)

    bin_volume = defaultdict(float)
    for idx, row in hist.iterrows():
        price = float(row["close"])
        vol = float(row.get("volume", 0.0) or 0.0)
        b = int(np.digitize(price, edges) - 1)
        if 0 <= b < bin_count:
            bin_volume[b] += vol

    if not bin_volume:
        return {
            "poc": None,
            "vah": None,
            "val": None,
            "high_volume_nodes": [],
            "low_volume_nodes": [],
            "total_volume": 0.0,
            "value_area_volume_percentage": 0.0,
            "value_area_percentage": value_area_percentage,
            "volume_quality": "INSUFFICIENT_DATA",
            "bin_count": 0,
        }

    total_volume = float(sum(bin_volume.values()))
    if total_volume <= 0:
        return {
            "poc": None,
            "vah": None,
            "val": None,
            "high_volume_nodes": [],
            "low_volume_nodes": [],
            "total_volume": 0.0,
            "value_area_volume_percentage": 0.0,
            "value_area_percentage": value_area_percentage,
            "volume_quality": "INSUFFICIENT_DATA",
            "bin_count": bin_count,
        }

    bin_prices = [(edges[i] + edges[i + 1]) / 2.0 for i in range(bin_count)]
    levels = [(idx, bin_prices[idx], vol) for idx, vol in bin_volume.items()]
    poc_index, poc_price, _ = max(levels, key=lambda x: x[2])
    poc = poc_price

    sorted_levels = sorted(levels, key=lambda item: item[1])
    lower = []
    upper = []
    total_va = 0.0
    target = total_volume * value_area_percentage
    running = 0.0
    expanded = set()

    for idx, price, vol in sorted_levels:
        if idx == poc_index:
            running += vol
            expanded.add(idx)
            continue
        lower.append((idx, price, vol))
        upper.append((idx, price, vol))

    for item in sorted(levels, key=lambda x: abs(x[1] - poc)):
        idx, price, vol = item
        if idx == poc_index:
            continue
        if price < poc:
            lower.append(item)
        else:
            upper.append(item)

    all_candidates = sorted([(idx, price, vol) for idx, price, vol in levels if idx != poc_index], key=lambda item: abs(item[1] - poc))
    selected = [poc_index]
    running = bin_volume[poc_index]
    flags = {poc_index}
    while running < target and all_candidates:
        choose_lower = False
        choose_upper = False
        if all_candidates:
            next_lower = min((item for item in all_candidates if item[1] < poc), key=lambda it: abs(it[1] - poc), default=None)
            next_upper = min((item for item in all_candidates if item[1] > poc), key=lambda it: abs(it[1] - poc), default=None)
            if next_lower is not None and next_upper is not None:
                choose_lower = next_lower[2] >= next_upper[2]
                choose_upper = not choose_lower
            elif next_lower is not None:
                choose_lower = True
            elif next_upper is not None:
                choose_upper = True

        if choose_lower and next_lower is not None and next_lower[0] not in flags:
            selected.append(next_lower[0])
            flags.add(next_lower[0])
            running += next_lower[2]
        elif choose_upper and next_upper is not None and next_upper[0] not in flags:
            selected.append(next_upper[0])
            flags.add(next_upper[0])
            running += next_upper[2]
        else:
            remaining = [item for item in all_candidates if item[0] not in flags]
            if not remaining:
                break
            picked = remaining[0]
            selected.append(picked[0])
            flags.add(picked[0])
            running += picked[2]
        all_candidates = [item for item in all_candidates if item[0] not in flags]

    selected_prices = sorted([bin_prices[idx] for idx in selected])
    vah = max(selected_prices) if selected_prices else None
    val = min(selected_prices) if selected_prices else None

    hvn = sorted([bin_prices[idx] for idx, vol in bin_volume.items() if vol >= (total_volume / len(bin_volume))], reverse=True)
    lvn = sorted([bin_prices[idx] for idx, vol in bin_volume.items() if vol < (total_volume / len(bin_volume))])

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "high_volume_nodes": hvn,
        "low_volume_nodes": lvn,
        "total_volume": total_volume,
        "value_area_volume_percentage": min(1.0, max(0.0, running / total_volume)) if total_volume else 0.0,
        "value_area_percentage": value_area_percentage,
        "volume_quality": "SUFFICIENT" if total_volume > 0 else "INSUFFICIENT_DATA",
        "bin_count": bin_count,
    }


def detect_volume_profile_reversion(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    if df.empty or profile.get("poc") is None:
        return {"strategy": "VALUE_AREA_REVERSION", "direction": "NONE", "setup_strength": 0.0, "reason": "INSUFFICIENT_DATA"}

    latest = df.iloc[-1]
    price = float(latest["close"])
    poc = float(profile["poc"])
    vah = float(profile["vah"]) if profile.get("vah") is not None else price
    val = float(profile["val"]) if profile.get("val") is not None else price

    if price > vah and price > poc:
        return {"strategy": "VALUE_AREA_REVERSION", "direction": "SHORT", "entry_reference": price, "stop_reference": vah, "target_reference": poc, "setup_strength": 0.5, "reason": "PRICE_ABOVE_VAH"}
    if price < val and price < poc:
        return {"strategy": "VALUE_AREA_REVERSION", "direction": "LONG", "entry_reference": price, "stop_reference": val, "target_reference": poc, "setup_strength": 0.5, "reason": "PRICE_BELOW_VAL"}
    return {"strategy": "VALUE_AREA_REVERSION", "direction": "NONE", "entry_reference": price, "stop_reference": None, "target_reference": poc, "setup_strength": 0.0, "reason": "NO_REVERSION"}


def detect_value_area_reversion(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    return detect_volume_profile_reversion(df, profile)
