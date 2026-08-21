import pandas as pd

from backend.engine.mtf_indicator_engine import MTFIndicatorEngine
from backend.engine.strategy_interpreter.parser import StrategyParser
from backend.engine.strategy_interpreter.evaluator.condition_evaluator import (
    ConditionEvaluator,
)
from backend.engine.backtester import Backtester


SYMBOL = "EURUSD.m"

RISK_REWARD = 2.0
STOP_DISTANCE = 0.0010


BUY_STRATEGY = (
    "BUY when "
    "H4 EMA_20 crosses above EMA_50 and "
    "H1 ADX > 20 and "
    "M30 RSI > 50 and "
    "M15 EMA_20 crosses above EMA_50 and "
    "M5 RSI > 50"
)


SELL_STRATEGY = (
    "SELL when "
    "H4 EMA_20 crosses below EMA_50 and "
    "H1 ADX > 20 and "
    "M30 RSI < 50 and "
    "M15 EMA_20 crosses below EMA_50 and "
    "M5 RSI < 50"
)


def align_condition_to_m15(
    condition,
    timeframe_data,
    m15_data,
):

    timeframe = getattr(
        condition,
        "timeframe",
        None,
    )

    if timeframe is None:
        timeframe = "M15"

    timeframe = timeframe.upper()

    source = timeframe_data[timeframe].copy()

    evaluator = ConditionEvaluator(source)

    signal = evaluator.evaluate_condition(
        condition
    )

    result = pd.DataFrame(
        {
            "timestamp": source["timestamp"],
            "signal": signal.astype(bool),
        }
    )

    result = result.sort_values(
        "timestamp"
    )

    target = m15_data[
        ["timestamp"]
    ].copy()

    target = target.sort_values(
        "timestamp"
    )

    aligned = pd.merge_asof(
        target,
        result,
        on="timestamp",
        direction="backward",
    )

    return aligned[
        "signal"
    ].fillna(False)


def generate_signals(
    strategy,
    data,
    m15,
):

    final_signal = pd.Series(
        True,
        index=m15.index,
        dtype=bool,
    )

    for condition in strategy.entry_conditions:

        timeframe = getattr(
            condition,
            "timeframe",
            "M15",
        )

        print(
            f"  Evaluating {timeframe}: "
            f"{condition.indicator} "
            f"{condition.operator} "
            f"{condition.value}"
        )

        aligned = align_condition_to_m15(
            condition,
            data,
            m15,
        )

        final_signal &= aligned.values

    return final_signal


def run_strategy(
    strategy_text,
    data,
    m15,
):

    strategy = StrategyParser().parse(
        strategy_text
    )

    direction = strategy.direction.lower()

    print()
    print(
        f"Strategy: {direction.upper()}"
    )

    print(
        f"Conditions: "
        f"{len(strategy.entry_conditions)}"
    )

    print()
    print(
        "Generating MTF signals..."
    )

    signal = generate_signals(
        strategy,
        data,
        m15,
    )

    signal_count = int(
        signal.sum()
    )

    print()

    if direction == "long":

        print(
            f"BUY signals: "
            f"{signal_count:,}"
        )

        backtest_direction = "BUY"

    else:

        print(
            f"SELL signals: "
            f"{signal_count:,}"
        )

        backtest_direction = "SELL"

    print()

    print(
        f"Running "
        f"{backtest_direction} backtest..."
    )

    backtester = Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    )

    trades = backtester.run(
        m15,
        signal,
        direction=backtest_direction,
    )

    return (
        strategy,
        trades,
        signal,
    )


def main():

    print("=" * 60)
    print("NEXA FUNDS AI")
    print("HISTORICAL STRATEGY BACKTEST")
    print("=" * 60)

    print()
    print("Loading indicators...")

    engine = MTFIndicatorEngine()

    data = engine.build(
        SYMBOL,
        [
            "H4",
            "H1",
            "M30",
            "M15",
            "M5",
        ],
    )

    print("Indicators loaded.")

    m15 = (
        data["M15"]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"M15 candles: "
        f"{len(m15):,}"
    )

    # ======================================================
    # BUY
    # ======================================================

    print()
    print("=" * 60)
    print("BUY STRATEGY")
    print("=" * 60)

    (
        buy_strategy,
        buy_trades,
        buy_signals,
    ) = run_strategy(
        BUY_STRATEGY,
        data,
        m15,
    )

    print()
    print("BUY RESULTS")

    Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    ).report(
        buy_trades
    )

    # ======================================================
    # SELL
    # ======================================================

    print()
    print("=" * 60)
    print("SELL STRATEGY")
    print("=" * 60)

    (
        sell_strategy,
        sell_trades,
        sell_signals,
    ) = run_strategy(
        SELL_STRATEGY,
        data,
        m15,
    )

    print()
    print("SELL RESULTS")

    Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    ).report(
        sell_trades
    )

    # ======================================================
    # SUMMARY
    # ======================================================

    total_trades = (
        len(buy_trades)
        + len(sell_trades)
    )

    total_r = (
        sum(
            trade.pnl
            for trade in buy_trades
        )
        +
        sum(
            trade.pnl
            for trade in sell_trades
        )
    )

    print()
    print("=" * 60)
    print("NEXA FUNDS AI SUMMARY")
    print("=" * 60)

    print(
        f"BUY trades:   "
        f"{len(buy_trades):,}"
    )

    print(
        f"SELL trades:  "
        f"{len(sell_trades):,}"
    )

    print(
        f"TOTAL trades: "
        f"{total_trades:,}"
    )

    print(
        f"TOTAL NET R:  "
        f"{total_r:.2f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()