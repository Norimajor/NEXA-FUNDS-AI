from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import OptionContract


def compute_iv_statistics(contracts: Iterable[OptionContract]) -> dict:
    normalized = list(contracts)
    if not normalized:
        return {
            "atm_iv": None,
            "iv_by_strike": {},
            "iv_by_expiration": {},
        }

    atm_values = []
    by_strike = defaultdict(list)
    by_expiration = defaultdict(list)

    for contract in normalized:
        if contract.implied_volatility is None and contract.iv is not None:
            contract.implied_volatility = contract.iv
        if contract.implied_volatility is None:
            continue
        if contract.underlying_price is not None and contract.strike is not None:
            diff = abs(contract.underlying_price - contract.strike)
            if diff <= max(1.0, contract.underlying_price * 0.01):
                atm_values.append(contract.implied_volatility)
        if contract.strike is not None:
            by_strike[contract.strike].append(contract.implied_volatility)
        if contract.expiration is not None:
            by_expiration[contract.expiration].append(contract.implied_volatility)

    atm_iv = sum(atm_values) / len(atm_values) if atm_values else None
    iv_by_strike = {strike: sum(values) / len(values) for strike, values in by_strike.items()}
    iv_by_expiration = {expiration: sum(values) / len(values) for expiration, values in by_expiration.items()}

    return {
        "atm_iv": atm_iv,
        "iv_by_strike": dict(sorted(iv_by_strike.items())),
        "iv_by_expiration": dict(sorted(iv_by_expiration.items())),
    }
