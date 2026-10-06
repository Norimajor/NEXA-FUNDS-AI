from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from market_data.models import OptionContract
from .models import OptionsFeatureVector


class OptionsFeatureEngine:
    """Create deterministic feature vectors from normalized option contracts."""

    def build(self, contracts: Iterable[OptionContract], asset: str = "BTC", reference_price: float | None = None) -> OptionsFeatureVector:
        contract_list = [contract for contract in contracts if contract is not None]
        reference = reference_price if reference_price is not None else self._resolve_reference_price(contract_list)

        total_call_oi = self._sum_metric(contract_list, "call", "open_interest")
        total_put_oi = self._sum_metric(contract_list, "put", "open_interest")
        total_oi = (total_call_oi or 0.0) + (total_put_oi or 0.0)
        put_call_oi_ratio = None
        if total_call_oi not in (None, 0):
            put_call_oi_ratio = (total_put_oi or 0.0) / total_call_oi
        elif total_put_oi not in (None, 0):
            put_call_oi_ratio = total_put_oi

        call_oi_concentration = (total_call_oi / total_oi) if total_oi else None
        put_oi_concentration = (total_put_oi / total_oi) if total_oi else None

        gamma_exposure = self._sum_metric(contract_list, None, "gamma")
        vega_exposure = self._sum_metric(contract_list, None, "vega")
        theta_exposure = self._sum_metric(contract_list, None, "theta")
        delta_exposure = self._sum_metric(contract_list, None, "delta")

        gamma_concentration_by_strike = self._group_by_strike(contract_list, "gamma")
        gamma_concentration_by_expiration = self._group_by_expiration(contract_list, "gamma")
        oi_concentration_by_strike = self._group_by_strike(contract_list, "open_interest")
        oi_concentration_by_expiration = self._group_by_expiration(contract_list, "open_interest")

        distance_to_major_oi_strike = self._distance_to_major_strike(contract_list, reference, "open_interest")
        distance_to_major_gamma_strike = self._distance_to_major_strike(contract_list, reference, "gamma")
        expiration_concentration = self._expiration_concentration(contract_list)
        near_term_oi_concentration, longer_term_oi_concentration = self._term_distribution(contract_list)
        call_put_greek_imbalance = self._call_put_greek_imbalance(contract_list)
        atm_near_atm_greek_exposure = self._atm_near_atm_greek_exposure(contract_list, reference)

        missing_fields = self._missing_fields(contract_list)
        return OptionsFeatureVector(
            asset=asset.upper(),
            timestamp=datetime.now(timezone.utc),
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            put_call_oi_ratio=put_call_oi_ratio,
            call_oi_concentration=call_oi_concentration,
            put_oi_concentration=put_oi_concentration,
            gamma_exposure=gamma_exposure,
            vega_exposure=vega_exposure,
            theta_exposure=theta_exposure,
            delta_exposure=delta_exposure,
            gamma_concentration_by_strike=gamma_concentration_by_strike,
            gamma_concentration_by_expiration=gamma_concentration_by_expiration,
            oi_concentration_by_strike=oi_concentration_by_strike,
            oi_concentration_by_expiration=oi_concentration_by_expiration,
            distance_to_major_oi_strike=distance_to_major_oi_strike,
            distance_to_major_gamma_strike=distance_to_major_gamma_strike,
            expiration_concentration=expiration_concentration,
            near_term_oi_concentration=near_term_oi_concentration,
            longer_term_oi_concentration=longer_term_oi_concentration,
            call_put_greek_imbalance=call_put_greek_imbalance,
            atm_near_atm_greek_exposure=atm_near_atm_greek_exposure,
            market_positioning_features={
                "call_oi": total_call_oi,
                "put_oi": total_put_oi,
                "delta_exposure": delta_exposure,
                "gamma_exposure": gamma_exposure,
                "vega_exposure": vega_exposure,
                "theta_exposure": theta_exposure,
                "put_call_oi_ratio": put_call_oi_ratio,
            },
            data_quality={"sufficient": self._sufficient_data(contract_list, reference), "missing_fields": missing_fields},
            missing_fields=missing_fields,
            volume=None,
            iv=None,
        )

    def _sum_metric(self, contracts: list[OptionContract], option_type: str | None, metric: str) -> float | None:
        values = []
        for contract in contracts:
            if option_type is not None and (contract.option_type or "").lower() != option_type:
                continue
            value = getattr(contract, metric, None)
            if value is not None:
                values.append(float(value))
        return sum(values) if values else None

    def _resolve_reference_price(self, contracts: list[OptionContract]) -> float | None:
        refs = [float(c.underlying_price) for c in contracts if c.underlying_price is not None]
        return max(refs) if refs else None

    def _group_by_strike(self, contracts: list[OptionContract], metric: str) -> dict[float, float] | None:
        grouped: dict[float, float] = defaultdict(float)
        for contract in contracts:
            if contract.strike is None:
                continue
            value = getattr(contract, metric, None)
            if value is not None:
                grouped[float(contract.strike)] += float(value)
        return dict(sorted(grouped.items())) if grouped else None

    def _group_by_expiration(self, contracts: list[OptionContract], metric: str) -> dict[str, float] | None:
        grouped: dict[str, float] = defaultdict(float)
        for contract in contracts:
            if contract.expiration is None:
                continue
            value = getattr(contract, metric, None)
            if value is not None:
                grouped[str(contract.expiration)] += float(value)
        return dict(sorted(grouped.items())) if grouped else None

    def _distance_to_major_strike(self, contracts: list[OptionContract], reference: float | None, metric: str) -> float | None:
        if reference is None:
            return None
        valid = [float(c.strike) for c in contracts if c.strike is not None and getattr(c, metric, None) is not None]
        if not valid:
            return None
        return abs(min(valid, key=lambda strike: abs(float(strike) - float(reference))) - float(reference))

    def _expiration_concentration(self, contracts: list[OptionContract]) -> dict[str, float] | None:
        grouped: dict[str, float] = defaultdict(float)
        total = 0.0
        for contract in contracts:
            if contract.expiration is None:
                continue
            oi = float(contract.open_interest) if contract.open_interest is not None else 0.0
            grouped[str(contract.expiration)] += oi
            total += oi
        if not grouped:
            return None
        return {expiry: value / total if total else 0.0 for expiry, value in sorted(grouped.items())}

    def _term_distribution(self, contracts: list[OptionContract]) -> tuple[float | None, float | None]:
        near_oi = 0.0
        long_oi = 0.0
        for contract in contracts:
            oi = float(contract.open_interest) if contract.open_interest is not None else 0.0
            if contract.days_to_expiration is not None and contract.days_to_expiration <= 7:
                near_oi += oi
            else:
                long_oi += oi
        total = near_oi + long_oi
        if total == 0:
            return None, None
        return near_oi / total, long_oi / total

    def _call_put_greek_imbalance(self, contracts: list[OptionContract]) -> float | None:
        call_total = 0.0
        put_total = 0.0
        found = False
        for contract in contracts:
            if contract.option_type is None or contract.delta is None:
                continue
            if contract.option_type.lower() == "call":
                call_total += float(contract.delta)
            elif contract.option_type.lower() == "put":
                put_total += abs(float(contract.delta))
            found = True
        return (call_total - put_total) if found else None

    def _atm_near_atm_greek_exposure(self, contracts: list[OptionContract], reference: float | None) -> float | None:
        if reference is None:
            return None
        values = []
        for contract in contracts:
            if contract.strike is None:
                continue
            if abs(float(contract.strike) - float(reference)) / max(float(reference), 1.0) > 0.02:
                continue
            if contract.gamma is not None:
                values.append(float(contract.gamma))
            elif contract.delta is not None:
                values.append(float(contract.delta))
        return sum(values) if values else None

    def _missing_fields(self, contracts: list[OptionContract]) -> list[str]:
        if not contracts:
            return ["underlying_price", "strike", "option_type", "open_interest", "expiration", "timestamp"]
        missing = []
        required = ["underlying_price", "strike", "option_type", "open_interest", "expiration", "timestamp"]
        for field_name in required:
            if all(getattr(contract, field_name, None) is None for contract in contracts):
                missing.append(field_name)
        return missing

    def _sufficient_data(self, contracts: list[OptionContract], reference: float | None) -> bool:
        if not contracts:
            return False
        if reference is None:
            return False
        valid_contracts = sum(1 for c in contracts if c.strike is not None and c.option_type is not None and c.expiration is not None)
        if valid_contracts < 3:
            return False
        if all(c.open_interest is None and c.gamma is None and c.delta is None and c.vega is None and c.theta is None for c in contracts):
            return False
        return True
