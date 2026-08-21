from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


@dataclass
class MarketData:
    symbol: str
    timeframe: str
    dataframe: pd.DataFrame


class MarketDataEngine:

    def __init__(self):
        self.required_columns = REQUIRED_COLUMNS

    def load_csv(
        self,
        file_path: str,
        symbol: str,
        timeframe: str,
    ) -> MarketData:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Market data file not found: {file_path}"
            )

        df = pd.read_csv(path)

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        missing = [
            column
            for column in self.required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
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
        ]

        for column in numeric_columns:
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

        df = df.sort_values("timestamp")

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )

        df = df.reset_index(drop=True)

        if "volume" not in df.columns:
            df["volume"] = 0.0

        if "spread" not in df.columns:
            df["spread"] = 0.0

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        ).fillna(0.0)

        df["spread"] = pd.to_numeric(
            df["spread"],
            errors="coerce",
        ).fillna(0.0)

        return MarketData(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            dataframe=df,
        )

    def validate(self, market_data: MarketData) -> dict:

        df = market_data.dataframe

        errors = []

        if df.empty:
            errors.append("Dataset is empty.")

        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        invalid_count = int(invalid_ohlc.sum())

        if invalid_count > 0:
            errors.append(
                f"{invalid_count} candles contain invalid OHLC values."
            )

        duplicate_count = int(
            df["timestamp"].duplicated().sum()
        )

        if duplicate_count > 0:
            errors.append(
                f"{duplicate_count} duplicate timestamps found."
            )

        return {
            "valid": len(errors) == 0,
            "symbol": market_data.symbol,
            "timeframe": market_data.timeframe,
            "rows": len(df),
            "start": (
                str(df["timestamp"].iloc[0])
                if not df.empty
                else None
            ),
            "end": (
                str(df["timestamp"].iloc[-1])
                if not df.empty
                else None
            ),
            "errors": errors,
        }
