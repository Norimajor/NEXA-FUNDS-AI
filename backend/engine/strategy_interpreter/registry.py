from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    category: str
    aliases: tuple[str, ...]
    operators: tuple[str, ...]


class ConditionRegistry:

    def __init__(self):

        self.conditions = {

            "EMA": ConditionSpec(
                "EMA",
                "trend",
                ("ema", "exponential moving average"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "SMA": ConditionSpec(
                "SMA",
                "trend",
                ("sma", "simple moving average"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "WMA": ConditionSpec(
                "WMA",
                "trend",
                ("wma", "weighted moving average"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "HMA": ConditionSpec(
                "HMA",
                "trend",
                ("hma", "hull moving average"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "DEMA": ConditionSpec(
                "DEMA",
                "trend",
                ("dema",),
                (">", "<", "cross_above", "cross_below"),
            ),

            "TEMA": ConditionSpec(
                "TEMA",
                "trend",
                ("tema",),
                (">", "<", "cross_above", "cross_below"),
            ),

            "VWMA": ConditionSpec(
                "VWMA",
                "trend",
                ("vwma",),
                (">", "<", "cross_above", "cross_below"),
            ),

            "ADX": ConditionSpec(
                "ADX",
                "trend",
                ("adx", "average directional index"),
                (">", "<"),
            ),

            "RSI": ConditionSpec(
                "RSI",
                "momentum",
                ("rsi", "relative strength index"),
                (">", "<"),
            ),

            "CCI": ConditionSpec(
                "CCI",
                "momentum",
                ("cci", "commodity channel index"),
                (">", "<"),
            ),

            "STOCHASTIC": ConditionSpec(
                "STOCHASTIC",
                "momentum",
                ("stochastic", "stoch"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "ROC": ConditionSpec(
                "ROC",
                "momentum",
                ("roc", "rate of change"),
                (">", "<"),
            ),

            "MOMENTUM": ConditionSpec(
                "MOMENTUM",
                "momentum",
                ("momentum",),
                (">", "<"),
            ),

            "TSI": ConditionSpec(
                "TSI",
                "momentum",
                ("tsi", "true strength index"),
                (">", "<"),
            ),

            "WILLIAMS_R": ConditionSpec(
                "WILLIAMS_R",
                "momentum",
                ("williams r", "williams %r"),
                (">", "<"),
            ),

            "ATR": ConditionSpec(
                "ATR",
                "volatility",
                ("atr", "average true range"),
                (">", "<"),
            ),

            "VWAP": ConditionSpec(
                "VWAP",
                "volume",
                ("vwap", "volume weighted average price"),
                (">", "<", "cross_above", "cross_below"),
            ),

            "OBV": ConditionSpec(
                "OBV",
                "volume",
                ("obv", "on balance volume"),
                (">", "<"),
            ),

            "MFI": ConditionSpec(
                "MFI",
                "volume",
                ("mfi", "money flow index"),
                (">", "<"),
            ),

            "CMF": ConditionSpec(
                "CMF",
                "volume",
                ("cmf", "chaikin money flow"),
                (">", "<"),
            ),

            "RELATIVE_VOLUME": ConditionSpec(
                "RELATIVE_VOLUME",
                "volume",
                ("relative volume", "relative_volume"),
                (">", "<"),
            ),

            "DOJI": ConditionSpec(
                "DOJI",
                "price_action",
                ("doji",),
                ("true", "false"),
            ),

            "PIN_BAR": ConditionSpec(
                "PIN_BAR",
                "price_action",
                ("pin bar", "pinbar"),
                ("true", "false"),
            ),

            "BULLISH_ENGULFING": ConditionSpec(
                "BULLISH_ENGULFING",
                "price_action",
                ("bullish engulfing",),
                ("true", "false"),
            ),

            "BEARISH_ENGULFING": ConditionSpec(
                "BEARISH_ENGULFING",
                "price_action",
                ("bearish engulfing",),
                ("true", "false"),
            ),

            "INSIDE_BAR": ConditionSpec(
                "INSIDE_BAR",
                "price_action",
                ("inside bar",),
                ("true", "false"),
            ),

            "BOS": ConditionSpec(
                "BOS",
                "smc",
                ("bos", "break of structure"),
                ("bullish", "bearish"),
            ),

            "CHOCH": ConditionSpec(
                "CHOCH",
                "smc",
                ("choch", "change of character"),
                ("bullish", "bearish"),
            ),

            "FVG": ConditionSpec(
                "FVG",
                "smc",
                ("fvg", "fair value gap"),
                ("bullish", "bearish"),
            ),

            "ORDER_BLOCK": ConditionSpec(
                "ORDER_BLOCK",
                "smc",
                ("order block", "order_block", "ob"),
                ("bullish", "bearish"),
            ),

            "LIQUIDITY_SWEEP": ConditionSpec(
                "LIQUIDITY_SWEEP",
                "smc",
                ("liquidity sweep", "liquidity_sweep", "sweep"),
                ("buy_side", "sell_side"),
            ),

            "PREMIUM_DISCOUNT": ConditionSpec(
                "PREMIUM_DISCOUNT",
                "smc",
                ("premium", "discount", "premium discount"),
                ("premium", "discount"),
            ),
        }

    def get(self, name: str):
        return self.conditions.get(name.upper())

    def all(self):
        return list(self.conditions.values())

    def by_category(self, category: str):
        return [
            condition
            for condition in self.conditions.values()
            if condition.category == category
        ]

    def find_by_alias(self, text: str):
        text = text.lower().strip()

        return [
            condition
            for condition in self.conditions.values()
            if text in condition.aliases
        ]

    def categories(self):
        return sorted({
            condition.category
            for condition in self.conditions.values()
        })
