import itertools
import re
import pandas as pd

from backend.engine.mtf_indicator_engine import MTFIndicatorEngine
from backend.engine.strategy_interpreter.parser import StrategyParser
from backend.engine.strategy_interpreter.evaluator.condition_evaluator import (
    ConditionEvaluator,
)
from backend.engine.backtester import Backtester


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "EURUSD.m"

RISK_REWARD = 2.0
STOP_DISTANCE = 0.0010

MIN_TRADES = 10
TOP_IMPROVEMENTS = 10

TIMEFRAMES = [
    "H4",
    "H1",
    "M30",
    "M15",
    "M5",
]


# ============================================================
# YOUR STRATEGY
# CHANGE THIS ONLY
# ============================================================

STRATEGY_TEXT = (
    "SELL when "
    "H4 EMA_20 crosses below EMA_50 and "
    "H1 ADX > 20 and "
    "M30 RSI < 50 and "
    "M15 EMA_20 crosses below EMA_50 and "
    "M5 RSI < 50"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(symbol):

    print()
    print("=" * 70)
    print("LOADING MARKET DATA")
    print("=" * 70)

    engine = MTFIndicatorEngine()

    data = engine.build(
        symbol,
        TIMEFRAMES,
    )

    m15 = (
        data["M15"]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"Symbol: {symbol}")
    print(f"M15 candles: {len(m15):,}")

    return data, m15


# ============================================================
# ALIGN CONDITION
# ============================================================

def align_condition(
    condition,
    data,
    m15,
):

    timeframe = getattr(
        condition,
        "timeframe",
        "M15",
    ).upper()

    source = data[timeframe].copy()

    evaluator = ConditionEvaluator(
        source
    )

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

    target = (
        m15[["timestamp"]]
        .copy()
        .sort_values("timestamp")
    )

    aligned = pd.merge_asof(
        target,
        result,
        on="timestamp",
        direction="backward",
    )

    return aligned["signal"].fillna(False)


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(
    strategy,
    data,
    m15,
):

    signal = pd.Series(
        True,
        index=m15.index,
        dtype=bool,
    )

    for condition in strategy.entry_conditions:

        aligned = align_condition(
            condition,
            data,
            m15,
        )

        signal &= aligned.values

    return signal


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_performance(trades):

    if not trades:
        return None

    wins = sum(
        trade.result == "WIN"
        for trade in trades
    )

    losses = sum(
        trade.result == "LOSS"
        for trade in trades
    )

    net_r = sum(
        trade.pnl
        for trade in trades
    )

    gross_profit = sum(
        trade.pnl
        for trade in trades
        if trade.pnl > 0
    )

    gross_loss = abs(
        sum(
            trade.pnl
            for trade in trades
            if trade.pnl < 0
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = 999.0

    win_rate = (
        wins /
        len(trades) *
        100
    )

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    losing_streak = 0
    max_losing_streak = 0

    for trade in trades:

        equity += trade.pnl

        peak = max(
            peak,
            equity,
        )

        drawdown = peak - equity

        max_drawdown = max(
            max_drawdown,
            drawdown,
        )

        if trade.result == "LOSS":

            losing_streak += 1

            max_losing_streak = max(
                max_losing_streak,
                losing_streak,
            )

        else:

            losing_streak = 0

    expectancy = (
        net_r /
        len(trades)
    )

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": net_r,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy,
        "max_losing_streak": max_losing_streak,
    }


# ============================================================
# BACKTEST
# ============================================================

def backtest_strategy(
    strategy,
    data,
    m15,
):

    signal = generate_signal(
        strategy,
        data,
        m15,
    )

    direction = strategy.direction.lower()

    backtester = Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    )

    trades = backtester.run(
        m15,
        signal,
        direction=("SELL" if direction.upper() == "SHORT" else "BUY"),
    )

    performance = calculate_performance(
        trades
    )

    return trades, performance


# ============================================================
# SCORE
# ============================================================

def score_performance(performance):

    if performance is None:
        return -9999.0

    if performance["trades"] < MIN_TRADES:
        return -9999.0

    return (
        performance["net_r"] * 5.0
        + performance["profit_factor"] * 5.0
        + performance["expectancy"] * 10.0
        - performance["max_drawdown"] * 1.5
    )


# ============================================================
# PARSE
# ============================================================

def parse_strategy(text):

    return StrategyParser().parse(
        text
    )


# ============================================================
# PRINT PERFORMANCE
# ============================================================

def print_performance(
    title,
    performance,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if performance is None:

        print("NO VALID TRADES")

        return

    print(
        f"Trades:             "
        f"{performance['trades']}"
    )

    print(
        f"Wins:               "
        f"{performance['wins']}"
    )

    print(
        f"Losses:             "
        f"{performance['losses']}"
    )

    print(
        f"Win Rate:           "
        f"{performance['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor:      "
        f"{performance['profit_factor']:.2f}"
    )

    print(
        f"Net R:              "
        f"{performance['net_r']:.2f}"
    )

    print(
        f"Expectancy:         "
        f"{performance['expectancy']:.3f}R"
    )

    print(
        f"Max Drawdown:       "
        f"{performance['max_drawdown']:.2f}R"
    )

    print(
        f"Max Losing Streak:  "
        f"{performance['max_losing_streak']}"
    )


# ============================================================
# CONDITION DESCRIPTION
# ============================================================

def describe_condition(condition):

    return (
        f"{condition.timeframe}: "
        f"{condition.indicator} "
        f"{condition.operator} "
        f"{condition.value}"
    )


# ============================================================
# EXTRACT NUMERIC VALUE
# ============================================================

def extract_number(text):

    match = re.search(
        r"[-+]?\d*\.?\d+",
        text,
    )

    if not match:
        return None

    return float(
        match.group()
    )


# ============================================================
# GENERATE IMPROVEMENT OPTIONS
# ============================================================

def generate_condition_variations(
    condition,
):

    timeframe = condition.timeframe
    indicator = condition.indicator
    operator = condition.operator

    value = extract_number(
        str(condition.value)
    )

    if value is None:
        return []

    variations = []

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if indicator == "RSI":

        if operator in (
            ">",
            "greater_than",
        ):

            values = [
                40,
                45,
                50,
                55,
                60,
                65,
            ]

            for new_value in values:

                if new_value != value:

                    variations.append(
                        f"{timeframe} RSI > {new_value}"
                    )

        elif operator in (
            "<",
            "less_than",
        ):

            values = [
                35,
                40,
                45,
                50,
                55,
                60,
            ]

            for new_value in values:

                if new_value != value:

                    variations.append(
                        f"{timeframe} RSI < {new_value}"
                    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    elif indicator == "ADX":

        if operator in (
            ">",
            "greater_than",
        ):

            values = [
                15,
                20,
                25,
                30,
                35,
            ]

            for new_value in values:

                if new_value != value:

                    variations.append(
                        f"{timeframe} ADX > {new_value}"
                    )

    # --------------------------------------------------------
    # EMA CROSS
    # --------------------------------------------------------

    elif indicator == "EMA_CROSS":

        if operator == "cross_above":

            variations.extend(
                [
                    f"{timeframe} EMA_20 crosses above EMA_100",
                    f"{timeframe} EMA_50 crosses above EMA_100",
                    f"{timeframe} EMA_20 crosses above EMA_200",
                    f"{timeframe} EMA_50 crosses above EMA_200",
                ]
            )

        elif operator == "cross_below":

            variations.extend(
                [
                    f"{timeframe} EMA_20 crosses below EMA_100",
                    f"{timeframe} EMA_50 crosses below EMA_100",
                    f"{timeframe} EMA_20 crosses below EMA_200",
                    f"{timeframe} EMA_50 crosses below EMA_200",
                ]
            )

    return variations


# ============================================================
# BUILD STRATEGY TEXT
# ============================================================

def build_strategy_text(
    strategy,
    replacements,
):

    direction = (
        strategy.direction.upper()
    )

    conditions = []

    for index, condition in enumerate(
        strategy.entry_conditions
    ):

        if index in replacements:

            conditions.append(
                replacements[index]
            )

        else:

            conditions.append(
                describe_condition(
                    condition
                )
            )

    return (
        f"{direction} when "
        + " and ".join(
            conditions
        )
    )


# ============================================================
# IMPROVEMENT SEARCH
# ============================================================

def find_improvements(
    strategy,
    baseline_score,
    data,
    m15,
):

    candidates = []

    print()
    print("=" * 70)
    print("SEARCHING FOR IMPROVEMENTS")
    print("=" * 70)

    for index, condition in enumerate(
        strategy.entry_conditions
    ):

        variations = (
            generate_condition_variations(
                condition
            )
        )

        if not variations:
            continue

        print(
            f"\nTesting alternatives for "
            f"condition #{index + 1}"
        )

        for variation in variations:

            try:

                text = build_strategy_text(
                    strategy,
                    {
                        index: variation
                    },
                )

                candidate = parse_strategy(
                    text
                )

                trades, performance = (
                    backtest_strategy(
                        candidate,
                        data,
                        m15,
                    )
                )

                if performance is None:
                    continue

                score = score_performance(
                    performance
                )

                candidates.append(
                    {
                        "score": score,
                        "text": text,
                        "performance": performance,
                        "changed_condition": (
                            index + 1
                        ),
                    }
                )

            except Exception as error:

                print(
                    f"Skipped: {error}"
                )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates


# ============================================================
# OOS SPLIT
# ============================================================

def split_oos(m15):

    split_index = int(
        len(m15) * 0.70
    )

    train = (
        m15
        .iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        m15
        .iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    return train, test


# ============================================================
# FILTER DATA BY RANGE
# ============================================================

def filter_data(
    data,
    start,
    end,
):

    filtered = {}

    for timeframe, frame in data.items():

        frame = frame.copy()

        timestamps = pd.to_datetime(
            frame["timestamp"]
        )

        mask = (
            (timestamps >= start)
            &
            (timestamps <= end)
        )

        filtered[timeframe] = (
            frame.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

    return filtered


# ============================================================
# OOS TEST
# ============================================================

def validate_oos(
    strategy_text,
    data,
    m15,
):

    train_m15, test_m15 = split_oos(
        m15
    )

    split_time = test_m15[
        "timestamp"
    ].iloc[0]

    test_data = filter_data(
        data,
        split_time,
        m15["timestamp"].iloc[-1],
    )

    strategy = parse_strategy(
        strategy_text
    )

    trades, performance = (
        backtest_strategy(
            strategy,
            test_data,
            test_m15,
        )
    )

    return performance


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEXA FUNDS AI")
    print("AI STRATEGY LAB")
    print("=" * 70)

    print()
    print("STRATEGY:")
    print(STRATEGY_TEXT)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data, m15 = load_data(
        SYMBOL
    )

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PARSING STRATEGY")
    print("=" * 70)

    strategy = parse_strategy(
        STRATEGY_TEXT
    )

    print(
        f"Direction: "
        f"{strategy.direction.upper()}"
    )

    print(
        f"Conditions: "
        f"{len(strategy.entry_conditions)}"
    )

    for condition in (
        strategy.entry_conditions
    ):

        print(
            " ",
            describe_condition(
                condition
            )
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    _, baseline = backtest_strategy(
        strategy,
        data,
        m15,
    )

    print_performance(
        "BASELINE BACKTEST",
        baseline,
    )

    baseline_score = score_performance(
        baseline
    )

    print(
        f"\nBaseline Score: "
        f"{baseline_score:.2f}"
    )

    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    improvements = find_improvements(
        strategy,
        baseline_score,
        data,
        m15,
    )

    print()
    print("=" * 70)
    print("TOP IMPROVEMENTS")
    print("=" * 70)

    if not improvements:

        print(
            "No valid improvements found."
        )

        return

    for number, result in enumerate(
        improvements[
            :TOP_IMPROVEMENTS
        ],
        start=1,
    ):

        p = result["performance"]

        print()
        print(
            f"#{number} "
            f"Score={result['score']:.2f}"
        )

        print(
            f"Trades:        "
            f"{p['trades']}"
        )

        print(
            f"Win Rate:      "
            f"{p['win_rate']:.2f}%"
        )

        print(
            f"Profit Factor: "
            f"{p['profit_factor']:.2f}"
        )

        print(
            f"Net R:         "
            f"{p['net_r']:.2f}"
        )

        print(
            f"Max DD:        "
            f"{p['max_drawdown']:.2f}R"
        )

        print(
            f"Changed condition: "
            f"#{result['changed_condition']}"
        )

        print(
            "Strategy:"
        )

        print(
            f"  {result['text']}"
        )

    # --------------------------------------------------------
    # OOS VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    candidates_to_validate = (
        improvements[
            :TOP_IMPROVEMENTS
        ]
    )

    validated = []

    for number, result in enumerate(
        candidates_to_validate,
        start=1,
    ):

        oos = validate_oos(
            result["text"],
            data,
            m15,
        )

        if oos is None:

            print(
                f"\n#{number}: "
                f"NO OOS TRADES"
            )

            continue

        print()
        print(
            f"#{number}"
        )

        print(
            f"Test Trades:   "
            f"{oos['trades']}"
        )

        print(
            f"Test Win Rate: "
            f"{oos['win_rate']:.2f}%"
        )

        print(
            f"Test PF:       "
            f"{oos['profit_factor']:.2f}"
        )

        print(
            f"Test Net R:    "
            f"{oos['net_r']:.2f}"
        )

        print(
            f"Test DD:       "
            f"{oos['max_drawdown']:.2f}R"
        )

        validated.append(
            {
                "number": number,
                "result": result,
                "oos": oos,
            }
        )

    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------

    robust = [
        item
        for item in validated
        if (
            item["oos"]["trades"] >= 3
            and
            item["oos"]["net_r"] > 0
            and
            item["oos"]["profit_factor"] > 1
        )
    ]

    print()
    print("=" * 70)
    print("NEXA FUNDS AI RECOMMENDATION")
    print("=" * 70)

    if robust:

        best = max(
            robust,
            key=lambda x: (
                x["oos"]["net_r"],
                x["oos"]["profit_factor"],
            ),
        )

        print()
        print(
            "BEST OOS-VALIDATED IMPROVEMENT"
        )

        print()
        print(
            best["result"]["text"]
        )

        print()
        print(
            f"OOS Trades: "
            f"{best['oos']['trades']}"
        )

        print(
            f"OOS Win Rate: "
            f"{best['oos']['win_rate']:.2f}%"
        )

        print(
            f"OOS Profit Factor: "
            f"{best['oos']['profit_factor']:.2f}"
        )

        print(
            f"OOS Net R: "
            f"{best['oos']['net_r']:.2f}"
        )

        print()
        print(
            "STATUS: "
            "PROMISING — REQUIRES FURTHER "
            "WALK-FORWARD AND MULTI-PAIR TESTING"
        )

    else:

        print()
        print(
            "No improvement survived the "
            "OOS validation strongly enough."
        )

        print()
        print(
            "The strategy should NOT be "
            "considered robust yet."
        )

    print()
    print("=" * 70)
    print("STRATEGY LAB COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
