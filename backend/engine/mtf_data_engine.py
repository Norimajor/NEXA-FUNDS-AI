from pathlib import Path
from typing import Dict

import pandas as pd


class MTFDataEngine:
    """
    Loads and aligns multi-timeframe market data.

    Supported timeframes:

        H4
        H1
        M30
        M15
        M5
        M1
    """

    TIMEFRAME_MINUTES = {
        "H4": 240,
        "H1": 60,
        "M30": 30,
        "M15": 15,
        "M5": 5,
        "M1": 1,
    }

    def __init__(
        self,
        data_directory: str = "data",
    ):

        self.data_directory = Path(
            data_directory
        )

    # ========================================================
    # FILE
    # ========================================================

    def get_file(
        self,
        symbol: str,
        timeframe: str,
    ) -> Path:

        symbol = symbol.upper()
        timeframe = timeframe.upper()

        filename = (
            f"{symbol.replace('.', '_')}"
            f"_{timeframe}.csv"
        )

        return (
            self.data_directory
            / filename
        )

    # ========================================================
    # LOAD ONE TIMEFRAME
    # ========================================================

    def load(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:

        timeframe = timeframe.upper()

        if timeframe not in self.TIMEFRAME_MINUTES:

            raise ValueError(
                f"Unsupported timeframe: "
                f"{timeframe}"
            )

        path = self.get_file(
            symbol,
            timeframe,
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Market data not found: {path}"
            )

        df = pd.read_csv(path)

        if "timestamp" not in df.columns:

            raise ValueError(
                f"{path} does not contain "
                f"a timestamp column."
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        df = df.sort_values(
            "timestamp"
        )

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )

        df = df.reset_index(
            drop=True
        )

        return df

    def validate_file(self, symbol: str, timeframe: str) -> dict:
        """Validate the raw CSV before load() normalizes it for indicators."""
        path = self.get_file(symbol, timeframe)
        report = {
            "status": "invalid",
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "source": str(path),
            "candles": 0,
            "start": None,
            "end": None,
            "missing_periods": 0,
            "duplicate_timestamps": 0,
            "invalid_candles": 0,
            "timezone": "naive",
            "errors": [],
        }
        if not path.exists():
            report["errors"].append("DATA_NOT_FOUND")
            return report

        raw = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close"}
        missing_columns = sorted(required.difference(raw.columns))
        if missing_columns:
            report["errors"].append(f"Missing required columns: {missing_columns}")
            return report

        timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
        numeric = raw[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        invalid_ohlc = (
            numeric.isna().any(axis=1)
            | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
            | (numeric["high"] < numeric["low"])
        )
        report["candles"] = int(len(raw))
        report["duplicate_timestamps"] = int(timestamps.duplicated().sum())
        report["invalid_candles"] = int((timestamps.isna() | invalid_ohlc).sum())
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            report["start"] = valid_timestamps.min().isoformat()
            report["end"] = valid_timestamps.max().isoformat()
            if getattr(valid_timestamps.dt, "tz", None) is not None:
                report["timezone"] = str(valid_timestamps.dt.tz)
            expected_minutes = self.TIMEFRAME_MINUTES.get(timeframe.upper())
            if expected_minutes and len(valid_timestamps) > 1:
                gaps = valid_timestamps.sort_values().diff().dropna()
                report["missing_periods"] = int((gaps > pd.Timedelta(minutes=expected_minutes * 1.5)).sum())

        if report["duplicate_timestamps"] or report["invalid_candles"]:
            report["errors"].append("DATA_QUALITY_FAILURE")
        else:
            report["status"] = "valid"
        return report

    # ========================================================
    # LOAD ALL AVAILABLE TIMEFRAMES
    # ========================================================

    def load_available(
        self,
        symbol: str,
    ) -> Dict[str, pd.DataFrame]:

        result = {}

        for timeframe in self.TIMEFRAME_MINUTES:

            path = self.get_file(
                symbol,
                timeframe,
            )

            if not path.exists():

                continue

            try:

                result[timeframe] = self.load(
                    symbol,
                    timeframe,
                )

            except Exception as error:

                print(
                    f"[WARNING] "
                    f"{symbol} {timeframe}: "
                    f"{error}"
                )

        return result

    # ========================================================
    # AVAILABLE TIMEFRAMES
    # ========================================================

    def available_timeframes(
        self,
        symbol: str,
    ):

        result = []

        for timeframe in self.TIMEFRAME_MINUTES:

            if self.get_file(
                symbol,
                timeframe,
            ).exists():

                result.append(
                    timeframe
                )

        return result

    # ========================================================
    # COMMON TIME RANGE
    # ========================================================

    def common_range(
        self,
        datasets: Dict[str, pd.DataFrame],
    ):

        if not datasets:

            return None, None

        starts = [
            df["timestamp"].min()
            for df in datasets.values()
            if not df.empty
        ]

        ends = [
            df["timestamp"].max()
            for df in datasets.values()
            if not df.empty
        ]

        if not starts or not ends:

            return None, None

        start = max(starts)
        end = min(ends)

        return start, end

    # ========================================================
    # ALIGN HIGHER TIMEFRAME DATA
    # ========================================================

    def align_to_base(
        self,
        base: pd.DataFrame,
        higher: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:

        timeframe = timeframe.upper()

        if higher.empty:

            return pd.DataFrame(
                index=base.index
            )

        base_sorted = (
            base
            .sort_values("timestamp")
            .copy()
        )

        higher_sorted = (
            higher
            .sort_values("timestamp")
            .copy()
        )

        # Rename OHLC columns so different
        # timeframes cannot overwrite each other.

        rename = {}

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
        ]:

            if column in higher_sorted.columns:

                rename[column] = (
                    f"{timeframe}_{column}"
                )

        higher_sorted = higher_sorted.rename(
            columns=rename
        )

        # Keep timestamp for merge_asof.

        merged = pd.merge_asof(
            base_sorted[
                ["timestamp"]
            ],
            higher_sorted,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

        merged = merged.drop(
            columns=["timestamp"]
        )

        merged.index = (
            base_sorted.index
        )

        return merged

    # ========================================================
    # BUILD MTF DATASET
    # ========================================================

    def build(
        self,
        symbol: str,
        base_timeframe: str = "M1",
    ) -> pd.DataFrame:

        base_timeframe = (
            base_timeframe.upper()
        )

        datasets = self.load_available(
            symbol
        )

        if base_timeframe not in datasets:

            raise FileNotFoundError(
                f"No {base_timeframe} data "
                f"available for {symbol}"
            )

        base = datasets[
            base_timeframe
        ].copy()

        base = base.sort_values(
            "timestamp"
        )

        base = base.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # ADD HIGHER TIMEFRAMES
        # ----------------------------------------------------

        for timeframe in self.TIMEFRAME_MINUTES:

            if timeframe == base_timeframe:

                continue

            if timeframe not in datasets:

                continue

            higher = datasets[
                timeframe
            ]

            aligned = self.align_to_base(
                base,
                higher,
                timeframe,
            )

            for column in aligned.columns:

                base[column] = aligned[
                    column
                ].values

        return base

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
        symbol: str,
    ):

        datasets = self.load_available(
            symbol
        )

        print()
        print("=" * 60)
        print(
            f"MTF DATA SUMMARY: {symbol}"
        )
        print("=" * 60)

        for timeframe in self.TIMEFRAME_MINUTES:

            if timeframe not in datasets:

                print(
                    f"{timeframe:<5} "
                    f"NOT AVAILABLE"
                )

                continue

            df = datasets[
                timeframe
            ]

            print(
                f"{timeframe:<5} "
                f"{len(df):>10,} candles  "
                f"{df['timestamp'].min()}  "
                f"->  "
                f"{df['timestamp'].max()}"
            )

        start, end = self.common_range(
            datasets
        )

        print()
        print(
            "COMMON RANGE:"
        )
        print(
            f"{start} -> {end}"
        )