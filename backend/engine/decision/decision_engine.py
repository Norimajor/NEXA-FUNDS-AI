from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DecisionResult:
    signal: str
    confidence: float

    bullish_score: float
    bearish_score: float

    reasons: List[str] = field(default_factory=list)

    trend: str = "UNKNOWN"
    market_regime: str = "UNKNOWN"
    market_condition: str = "UNKNOWN"
    volatility_regime: str = "UNKNOWN"

    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    risk_reward: float = 0.0

    details: Dict = field(default_factory=dict)


class DecisionEngine:
    """
    NEXA FUNDS AI Decision Engine

    Combines evidence from:
        - Market regime
        - Market structure
        - BOS / CHOCH
        - Liquidity
        - FVG
        - Order blocks
        - Premium / discount
        - Momentum
        - Volume
        - Price action
    """

    def __init__(
        self,
        minimum_score: float = 60.0,
        strong_score: float = 80.0,
        risk_reward: float = 2.0,
    ):
        self.minimum_score = minimum_score
        self.strong_score = strong_score
        self.risk_reward = risk_reward

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _is_true(value) -> bool:
        return bool(value) is True

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------------
    # MAIN DECISION
    # ---------------------------------------------------------

    def evaluate(self, data: Dict) -> DecisionResult:

        bullish = 0.0
        bearish = 0.0

        bullish_reasons = []
        bearish_reasons = []

        # =====================================================
        # 1. MARKET REGIME
        # =====================================================

        trend = str(data.get("TREND_REGIME", "UNKNOWN"))
        market_condition = str(
            data.get("MARKET_CONDITION", "UNKNOWN")
        )
        volatility = str(
            data.get("VOLATILITY_REGIME", "UNKNOWN")
        )
        market_regime = str(
            data.get("MARKET_REGIME", "UNKNOWN")
        )

        if trend in ("BULL", "STRONG_BULL"):
            bullish += 15

            bullish_reasons.append(
                f"Bullish trend regime: {trend}"
            )

        elif trend in ("BEAR", "STRONG_BEAR"):
            bearish += 15

            bearish_reasons.append(
                f"Bearish trend regime: {trend}"
            )

        # Trending market gets a small confirmation bonus.
        if market_condition == "TRENDING":
            if bullish > bearish:
                bullish += 5
                bullish_reasons.append(
                    "Market is trending"
                )
            elif bearish > bullish:
                bearish += 5
                bearish_reasons.append(
                    "Market is trending"
                )

        # =====================================================
        # 2. MARKET STRUCTURE
        # =====================================================

        high_structure = data.get("HIGH_STRUCTURE")
        low_structure = data.get("LOW_STRUCTURE")

        if high_structure == "HH":
            bullish += 5
            bullish_reasons.append(
                "Market structure shows Higher High"
            )

        elif high_structure == "LH":
            bearish += 5
            bearish_reasons.append(
                "Market structure shows Lower High"
            )

        if low_structure == "HL":
            bullish += 5
            bullish_reasons.append(
                "Market structure shows Higher Low"
            )

        elif low_structure == "LL":
            bearish += 5
            bearish_reasons.append(
                "Market structure shows Lower Low"
            )

        # =====================================================
        # 3. BOS / CHOCH
        # =====================================================

        if self._is_true(data.get("BULLISH_BOS")):
            bullish += 15
            bullish_reasons.append(
                "Bullish Break of Structure"
            )

        if self._is_true(data.get("BEARISH_BOS")):
            bearish += 15
            bearish_reasons.append(
                "Bearish Break of Structure"
            )

        if self._is_true(data.get("BULLISH_CHOCH")):
            bullish += 12
            bullish_reasons.append(
                "Bullish Change of Character"
            )

        if self._is_true(data.get("BEARISH_CHOCH")):
            bearish += 12
            bearish_reasons.append(
                "Bearish Change of Character"
            )

        # =====================================================
        # 4. LIQUIDITY
        # =====================================================

        if self._is_true(data.get("SELL_SIDE_SWEEP")):
            bullish += 10
            bullish_reasons.append(
                "Sell-side liquidity swept"
            )

        if self._is_true(data.get("BUY_SIDE_SWEEP")):
            bearish += 10
            bearish_reasons.append(
                "Buy-side liquidity swept"
            )

        # =====================================================
        # 5. FAIR VALUE GAP
        # =====================================================

        if self._is_true(data.get("BULLISH_FVG")):
            bullish += 10
            bullish_reasons.append(
                "Bullish Fair Value Gap detected"
            )

        if self._is_true(data.get("BEARISH_FVG")):
            bearish += 10
            bearish_reasons.append(
                "Bearish Fair Value Gap detected"
            )

        # =====================================================
        # 6. ORDER BLOCK
        # =====================================================

        if self._is_true(data.get("BULLISH_ORDER_BLOCK")):
            bullish += 10
            bullish_reasons.append(
                "Bullish Order Block detected"
            )

        if self._is_true(data.get("BEARISH_ORDER_BLOCK")):
            bearish += 10
            bearish_reasons.append(
                "Bearish Order Block detected"
            )

        # =====================================================
        # 7. PREMIUM / DISCOUNT
        # =====================================================

        if self._is_true(data.get("IN_DISCOUNT")):
            bullish += 10
            bullish_reasons.append(
                "Price is in discount"
            )

        if self._is_true(data.get("IN_PREMIUM")):
            bearish += 10
            bearish_reasons.append(
                "Price is in premium"
            )

        # =====================================================
        # 8. MOMENTUM
        # =====================================================

        rsi = data.get("RSI_14")

        if rsi is not None:
            rsi = self._safe_float(rsi)

            if rsi > 55:
                bullish += 5
                bullish_reasons.append(
                    f"RSI bullish ({rsi:.2f})"
                )

            elif rsi < 45:
                bearish += 5
                bearish_reasons.append(
                    f"RSI bearish ({rsi:.2f})"
                )

        adx = data.get("ADX_14")

        if adx is not None:
            adx = self._safe_float(adx)

            if adx >= 25:
                if bullish > bearish:
                    bullish += 5
                    bullish_reasons.append(
                        f"ADX confirms trend strength ({adx:.2f})"
                    )

                elif bearish > bullish:
                    bearish += 5
                    bearish_reasons.append(
                        f"ADX confirms trend strength ({adx:.2f})"
                    )

        # =====================================================
        # 9. VOLUME
        # =====================================================

        relative_volume = data.get("RELATIVE_VOLUME")

        if relative_volume is not None:

            relative_volume = self._safe_float(
                relative_volume
            )

            if relative_volume >= 1.2:

                if bullish > bearish:
                    bullish += 5
                    bullish_reasons.append(
                        f"Volume confirms bullish pressure "
                        f"({relative_volume:.2f}x)"
                    )

                elif bearish > bullish:
                    bearish += 5
                    bearish_reasons.append(
                        f"Volume confirms bearish pressure "
                        f"({relative_volume:.2f}x)"
                    )

        # =====================================================
        # FINAL SIGNAL
        # =====================================================

        total = bullish + bearish

        if total <= 0:

            signal = "NONE"
            confidence = 0.0

        else:

            dominant = max(bullish, bearish)

            # Confidence measures dominance, not simply raw score.
            confidence = (
                dominant / total
            ) * 100.0

            if bullish > bearish and bullish >= self.minimum_score:

                if bullish >= self.strong_score:
                    signal = "STRONG_BUY"
                else:
                    signal = "BUY"

            elif bearish > bullish and bearish >= self.minimum_score:

                if bearish >= self.strong_score:
                    signal = "STRONG_SELL"
                else:
                    signal = "SELL"

            else:
                signal = "NONE"

        # =====================================================
        # ENTRY / SL / TP
        # =====================================================

        entry = data.get("close")

        stop_loss = None
        take_profit = None

        if entry is not None:

            entry = self._safe_float(entry)

            atr = data.get("ATR")

            if atr is None:
                atr = data.get("REGIME_ATR")

            if atr is not None:

                atr = self._safe_float(atr)

                if signal in ("BUY", "STRONG_BUY"):

                    stop_loss = entry - atr

                    take_profit = (
                        entry
                        + atr * self.risk_reward
                    )

                elif signal in ("SELL", "STRONG_SELL"):

                    stop_loss = entry + atr

                    take_profit = (
                        entry
                        - atr * self.risk_reward
                    )

        # =====================================================
        # RESULT
        # =====================================================

        reasons = []

        if signal in ("BUY", "STRONG_BUY"):
            reasons = bullish_reasons

        elif signal in ("SELL", "STRONG_SELL"):
            reasons = bearish_reasons

        return DecisionResult(
            signal=signal,
            confidence=round(confidence, 2),

            bullish_score=round(bullish, 2),
            bearish_score=round(bearish, 2),

            reasons=reasons,

            trend=trend,
            market_regime=market_regime,
            market_condition=market_condition,
            volatility_regime=volatility,

            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,

            risk_reward=self.risk_reward,

            details={
                "bullish_reasons": bullish_reasons,
                "bearish_reasons": bearish_reasons,
            },
        )