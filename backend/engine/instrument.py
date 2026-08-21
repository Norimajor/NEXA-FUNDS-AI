from dataclasses import dataclass
from typing import Optional


@dataclass
class InstrumentSpec:
    symbol: str
    asset_class: str

    digits: int
    point: float

    tick_size: float
    tick_value: float

    contract_size: float

    volume_min: float
    volume_max: float
    volume_step: float

    pip_size: Optional[float] = None

    currency_base: Optional[str] = None
    currency_quote: Optional[str] = None


class InstrumentEngine:

    def __init__(self):
        self.instruments: dict[str, InstrumentSpec] = {}

    def register(self, instrument: InstrumentSpec):
        self.instruments[instrument.symbol.upper()] = instrument

    def get(self, symbol: str) -> InstrumentSpec:
        symbol = symbol.upper()

        if symbol not in self.instruments:
            raise ValueError(f"Instrument '{symbol}' is not registered.")

        return self.instruments[symbol]

    def calculate_price_distance(
        self,
        symbol: str,
        points: float
    ) -> float:
        instrument = self.get(symbol)
        return points * instrument.point

    def calculate_pip_distance(
        self,
        symbol: str,
        pips: float
    ) -> float:
        instrument = self.get(symbol)

        if instrument.pip_size is None:
            raise ValueError(
                f"{symbol} does not have a defined pip size."
            )

        return pips * instrument.pip_size

    def calculate_ticks(
        self,
        symbol: str,
        price_distance: float
    ) -> float:
        instrument = self.get(symbol)

        if instrument.tick_size <= 0:
            raise ValueError(
                f"Invalid tick size for {symbol}."
            )

        return price_distance / instrument.tick_size

    def calculate_tick_value(
        self,
        symbol: str,
        price_distance: float,
        volume: float
    ) -> float:
        instrument = self.get(symbol)

        ticks = self.calculate_ticks(
            symbol,
            price_distance
        )

        return ticks * instrument.tick_value * volume

    def list_instruments(self):
        return list(self.instruments.keys())
