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

        timeframe = timeframe.upper()

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
            )

        conditions = []

        # ====================================================
        # MOVING AVERAGES
        # ====================================================

        conditions.extend(
            self._parse_moving_averages(
                text,
                timeframe,
            )
        )

        # ====================================================
        # NUMERIC INDICATORS
        # ====================================================

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
                    timeframe,
                )
            )

        # ====================================================
        # DIRECTION
        # ====================================================

        direction = "both"

        has_buy = bool(
            re.search(
                r"\b(buy|long|bullish)\b",
                text,
                re.I,
            )
        )

        has_sell = bool(
            re.search(
                r"\b(sell|short|bearish)\b",
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
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_conditions=conditions,
        )

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
                else None
            )

            operator_text = (
                match.group("operator")
                .lower()
                .strip()
            )

            value = float(
                match.group("value")
            )

            if operator_text in {
                "above",
                "greater than",
                ">",
            }:

                operator = ">"

            elif operator_text in {
                "below",
                "less than",
                "<",
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
                )
            )

        return conditions