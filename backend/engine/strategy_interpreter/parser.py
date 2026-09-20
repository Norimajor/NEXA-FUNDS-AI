import re

from .models import (
    StrategyCondition,
    StrategyDefinition,
    SUPPORTED_TIMEFRAMES,
)


class StrategyParser:

    TIMEFRAMES = r"H4|H1|M30|M15|M5|M1"

    MA_TYPES = r"EMA|SMA|WMA|HMA|DEMA|TEMA"

    def parse(
        self,
        text: str,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
    ) -> StrategyDefinition:

        text = text.strip()

        if not text:
            raise ValueError(
                "Strategy text cannot be empty."
            )

        detected_symbol = self._extract_symbol(text, symbol)
        detected_timeframe = self._extract_timeframe(text, timeframe)

        detected_timeframe = detected_timeframe.upper()
        if detected_timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{detected_timeframe}'. "
                f"Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
            )

        conditions = []

        conditions.extend(
            self._parse_moving_averages(
                text,
                detected_timeframe,
            )
        )

        conditions.extend(
            self._parse_ema_crossover_alias(
                text,
                detected_timeframe,
            )
        )

        for indicator in [
            "RSI",
            "ADX",
            "CCI",
            "ROC",
            "MOMENTUM",
            "ATR",
            "TSI",
            "WILLIAMS_R",
            "VWAP",
            "OBV",
            "MFI",
            "CMF",
            "RELATIVE_VOLUME",
        ]:

            conditions.extend(
                self._parse_numeric_indicator(
                    text,
                    indicator,
                    detected_timeframe,
                )
            )

        direction = "both"

        has_buy = bool(
            re.search(
                r"\b(buy|long|bullish|uptrend)\b",
                text,
                re.I,
            )
        )

        has_sell = bool(
            re.search(
                r"\b(sell|short|bearish|downtrend)\b",
                text,
                re.I,
            )
        )

        if has_buy and not has_sell:
            direction = "long"

        elif has_sell and not has_buy:
            direction = "short"

        return StrategyDefinition(
            name="NEXA FUNDS AI Parsed Strategy",
            symbol=detected_symbol,
            timeframe=detected_timeframe,
            direction=direction,
            entry_conditions=conditions,
            allow_reentry=True,
            max_simultaneous_positions=1,
            pyramiding=False,
        )

    def _extract_symbol(self, text: str, default: str) -> str:
        symbols = (
            "XAUUSD",
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "NZDUSD",
            "USDCAD",
            "USDCHF",
            "EURJPY",
            "GBPJPY",
            "AUDJPY",
            "NAS100",
            "US30",
            "BTCUSD",
            "ETHUSD",
        )
        for symbol in symbols:
            if re.search(rf"\b{re.escape(symbol)}\b", text, re.I):
                return symbol
        return default.upper()

    def _extract_timeframe(self, text: str, default: str) -> str:
        normalized = text.upper()
        for code in ["H4", "H1", "M30", "M15", "M5", "M1"]:
            if re.search(rf"\b{code}\b", normalized):
                return code
        match = re.search(r"\b(\d{1,2})\s*(M|H)\b", text, re.I)
        if match:
            value = int(match.group(1))
            unit = match.group(2).upper()
            mapped = f"{unit}{value}"
            if mapped in SUPPORTED_TIMEFRAMES:
                return mapped
        if re.search(r"\b15m\b|\b1h\b|\b5m\b|\b30m\b|\b4h\b", text, re.I):
            lower = text.lower()
            if "15m" in lower or "m15" in lower:
                return "M15"
            if "30m" in lower or "m30" in lower:
                return "M30"
            if "5m" in lower or "m5" in lower:
                return "M5"
            if "1h" in lower or "h1" in lower:
                return "H1"
            if "4h" in lower or "h4" in lower:
                return "H4"
        return default.upper()

    def _parse_ema_crossover_alias(self, text, default_timeframe):
        conditions = []
        patterns = [
            r"(?:(?P<timeframe>H4|H1|M30|M15|M5|M1)\s+)?(?:EMA|EMA\s*)\s*(?P<fast>\d+)\s*(?:/|to|and)?\s*(?:EMA\s*)?(?P<slow>\d+)\s*(?:crossover|cross(?:es)?\s+(?:above|below))",
            r"(?:(?P<timeframe>H4|H1|M30|M15|M5|M1)\s+)?(?P<fast>\d+)\s*/\s*(?P<slow>\d+)\s*(?:EMA|moving average)\s*(?:crossover|cross(?:es)?\s+(?:above|below))",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                timeframe = (match.groupdict().get("timeframe") or default_timeframe).upper()
                fast_value = match.groupdict().get("fast")
                slow_value = match.groupdict().get("slow")
                if fast_value is None or slow_value is None:
                    continue
                fast = int(fast_value)
                slow = int(slow_value)
                lower = match.group(0).lower()
                operator = "cross_above" if "above" in lower or "bull" in lower else "cross_below"
                if "below" in lower or "bear" in lower:
                    operator = "cross_below"
                conditions.append(
                    StrategyCondition(
                        indicator="EMA_CROSS",
                        operator=operator,
                        value=slow,
                        period=fast,
                        timeframe=timeframe,
                    )
                )
        return conditions

    # ========================================================
    # MOVING AVERAGE CONDITIONS
    # ========================================================

    def _parse_moving_averages(
        self,
        text,
        default_timeframe,
    ):

        conditions = []

        # ----------------------------------------------------
        # CROSS ABOVE
        #
        # H4 EMA_20 crosses above EMA_50
        # EMA_20 crosses above EMA_50
        # ----------------------------------------------------

        pattern_above = re.compile(
            rf"""
            (?:
                (?P<timeframe>{self.TIMEFRAMES})
                \s+
            )?

            (?P<fast_type>{self.MA_TYPES})
            [_\s]*
            (?P<fast_period>\d+)

            \s*

            (?P<operator>
                crosses\s+above
                |
                cross\s+above
            )

            \s*

            (?P<slow_type>{self.MA_TYPES})
            [_\s]*
            (?P<slow_period>\d+)
            """,
            re.I | re.X,
        )

        for match in pattern_above.finditer(text):

            condition_timeframe = (
                match.group("timeframe")
                or default_timeframe
            ).upper()

            fast_type = (
                match.group("fast_type").upper()
            )

            slow_type = (
                match.group("slow_type").upper()
            )

            fast_period = int(
                match.group("fast_period")
            )

            slow_period = int(
                match.group("slow_period")
            )

            # The evaluator currently has a dedicated
            # EMA_CROSS implementation. For now, only
            # EMA/EMA crosses use that implementation.
            if fast_type != "EMA" or slow_type != "EMA":

                raise ValueError(
                    "Moving-average crosses currently "
                    "require EMA/EMA."
                )

            conditions.append(
                StrategyCondition(
                    indicator="EMA_CROSS",
                    operator="cross_above",
                    value=slow_period,
                    period=fast_period,
                    timeframe=condition_timeframe,
                )
            )

        # ----------------------------------------------------
        # CROSS BELOW
        # ----------------------------------------------------

        pattern_below = re.compile(
            rf"""
            (?:
                (?P<timeframe>{self.TIMEFRAMES})
                \s+
            )?

            (?P<fast_type>{self.MA_TYPES})
            [_\s]*
            (?P<fast_period>\d+)

            \s*

            (?P<operator>
                crosses\s+below
                |
                cross\s+below
            )

            \s*

            (?P<slow_type>{self.MA_TYPES})
            [_\s]*
            (?P<slow_period>\d+)
            """,
            re.I | re.X,
        )

        for match in pattern_below.finditer(text):

            condition_timeframe = (
                match.group("timeframe")
                or default_timeframe
            ).upper()

            fast_type = (
                match.group("fast_type").upper()
            )

            slow_type = (
                match.group("slow_type").upper()
            )

            fast_period = int(
                match.group("fast_period")
            )

            slow_period = int(
                match.group("slow_period")
            )

            if fast_type != "EMA" or slow_type != "EMA":

                raise ValueError(
                    "Moving-average crosses currently "
                    "require EMA/EMA."
                )

            conditions.append(
                StrategyCondition(
                    indicator="EMA_CROSS",
                    operator="cross_below",
                    value=slow_period,
                    period=fast_period,
                    timeframe=condition_timeframe,
                )
            )

        return conditions

    # ========================================================
    # NUMERIC INDICATORS
    # ========================================================

    def _parse_numeric_indicator(
        self,
        text,
        indicator,
        default_timeframe,
    ):

        conditions = []

        pattern = re.compile(
            rf"""
            (?:
                (?P<timeframe>{self.TIMEFRAMES})
                \s+
            )?

            \b{re.escape(indicator)}\b

            \s*

            (?P<period>\d+)?

            \s*

            (?P<operator>
                above
                |
                greater\s+than
                |
                >
                |
                below
                |
                less\s+than
                |
                <
                |
                is\s+above
                |
                is\s+below
                |
                is\s+greater\s+than
                |
                is\s+less\s+than
                |
                        drops\s+below
                        |
                        crosses\s+below
                        |
                        crosses\s+above
                        |
                        falls\s+below
                        |
                        declines\s+below
                        |
                        closes\s+below
                        |
                        rises\s+above
                        |
                        closes\s+above
                        |
                        stays\s+above
                        |
                        stays\s+below
            )

            \s*

            (?P<value>
                -?\d+(?:\.\d+)?
            )
            """,
            re.I | re.X,
        )

        for match in pattern.finditer(text):

            condition_timeframe = (
                match.group("timeframe")
                or default_timeframe
            ).upper()

            period_text = match.group("period")

            period = (
                int(period_text)
                if period_text
                else 14 if indicator == "RSI" else None
            )

            operator_text = (
                match.group("operator")
                .lower()
                .strip()
            )

            value = float(
                match.group("value")
            )

            entry_semantics = "cross" if "cross" in text[max(0, match.start() - 24):match.start()].lower() else "condition"
            if operator_text in {"drops below", "crosses below", "falls below", "declines below", "rises above", "crosses above", "closes below", "closes above"}:
                entry_semantics = "cross"

            if operator_text in {
                "above",
                "greater than",
                ">",
                "is above",
                "is greater than",
                "above 0",
                "rises above",
                "crosses above",
                "closes above",
                "stays above",
                "drops above",
            }:

                operator = ">"

            elif operator_text in {
                "below",
                "less than",
                "<",
                "is below",
                "is less than",
                "drops below",
                "crosses below",
                "falls below",
                "declines below",
                "closes below",
                "stays below",
            }:

                operator = "<"

            else:

                raise ValueError(
                    f"Unsupported operator: "
                    f"{operator_text}"
                )

            conditions.append(
                StrategyCondition(
                    indicator=indicator,
                    operator=operator,
                    value=value,
                    period=period,
                    timeframe=condition_timeframe,
                    entry_semantics=entry_semantics,
                )
            )

        return conditions