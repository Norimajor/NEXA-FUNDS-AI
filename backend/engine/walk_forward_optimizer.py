import itertools
import pandas as pd

from backend.engine.mtf_indicator_engine import MTFIndicatorEngine
from backend.engine.strategy_interpreter.parser import StrategyParser
from backend.engine.strategy_interpreter.evaluator.condition_evaluator import (
    ConditionEvaluator,
)
from backend.engine.backtester import Backtester


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "EURUSD.m"

RISK_REWARD = 2.0
STOP_DISTANCE = 0.0010

MIN_TRAIN_TRADES = 10

TOP_STRATEGIES = 5


# ============================================================
# CONDITION ALIGNMENT
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

    source = data[timeframe]

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

    target = m15[
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

    return aligned["signal"].fillna(
        False
    )


# ============================================================
# GENERATE SIGNAL
# ============================================================

def generate_signal(
    conditions,
    data,
    m15,
):

    signal = pd.Series(
        True,
        index=m15.index,
        dtype=bool,
    )

    for condition in conditions:

        aligned = align_condition(
            condition,
            data,
            m15,
        )

        signal &= aligned.values

    return signal


# ============================================================
# RUN BACKTEST
# ============================================================

def evaluate_strategy(
    conditions,
    direction,
    data,
    m15,
):

    signal = generate_signal(
        conditions,
        data,
        m15,
    )

    backtester = Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    )

    trades = backtester.run(
        m15,
        signal,
        direction=direction,
    )

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

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 999.0
    )

    win_rate = (
        wins / len(trades) * 100
    )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

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

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": net_r,
        "max_drawdown_r": max_drawdown,
        "conditions": conditions,
        "trades_data": trades,
    }


# ============================================================
# CONDITION POOLS
# ============================================================

def build_conditions(
    direction
):

    if direction == "BUY":

        return [

            # H4 trend
            [
                "H4 EMA_20 crosses above EMA_50",
                "H4 EMA_50 crosses above EMA_100",
                "H4 EMA_50 crosses above EMA_200",
            ],

            # H1 trend strength
            [
                "H1 ADX > 20",
                "H1 ADX > 25",
                "H1 ADX > 30",
            ],

            # M30 momentum
            [
                "M30 RSI > 50",
                "M30 RSI > 55",
                "M30 RSI > 60",
            ],

            # M15 confirmation
            [
                "M15 EMA_20 crosses above EMA_50",
                "M15 EMA_20 crosses above EMA_100",
                "M15 EMA_50 crosses above EMA_100",
            ],

            # M5 confirmation
            [
                "M5 RSI > 50",
                "M5 RSI > 55",
                "M5 RSI > 60",
            ],
        ]

    return [

        # H4 trend
        [
            "H4 EMA_20 crosses below EMA_50",
            "H4 EMA_50 crosses below EMA_100",
            "H4 EMA_50 crosses below EMA_200",
        ],

        # H1 trend strength
        [
            "H1 ADX > 20",
            "H1 ADX > 25",
            "H1 ADX > 30",
        ],

        # M30 momentum
        [
            "M30 RSI < 50",
            "M30 RSI < 45",
            "M30 RSI < 40",
        ],

        # M15 confirmation
        [
            "M15 EMA_20 crosses below EMA_50",
            "M15 EMA_20 crosses below EMA_100",
            "M15 EMA_50 crosses below EMA_100",
        ],

        # M5 confirmation
        [
            "M5 RSI < 50",
            "M5 RSI < 45",
            "M5 RSI < 40",
        ],
    ]


# ============================================================
# SCORE
# ============================================================

def score_result(result):

    if result is None:
        return -999999

    return (
        result["net_r"]
        + result["profit_factor"] * 2
        - result["max_drawdown_r"] * 0.25
        + result["win_rate"] * 0.02
    )


# ============================================================
# OPTIMIZE TRAINING PERIOD
# ============================================================

def optimize_training(
    direction,
    data,
    m15,
):

    parser = StrategyParser()

    pools = build_conditions(
        direction
    )

    combinations = itertools.product(
        *pools
    )

    results = []

    for combination in combinations:

        strategy_text = (
            f"{direction} when "
            + " and ".join(
                combination
            )
        )

        try:

            strategy = parser.parse(
                strategy_text
            )

            result = evaluate_strategy(
                strategy.entry_conditions,
                direction,
                data,
                m15,
            )

            if result is None:
                continue

            if (
                result["trades"]
                < MIN_TRAIN_TRADES
            ):
                continue

            result["strategy_text"] = (
                strategy_text
            )

            result["score"] = (
                score_result(result)
            )

            results.append(
                result
            )

        except Exception:
            continue

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# ============================================================
# PRINT STRATEGY
# ============================================================

def print_strategy(
    result
):

    print()

    for condition in result[
        "conditions"
    ]:

        print(
            f"  {condition.timeframe}: "
            f"{condition.indicator} "
            f"{condition.operator} "
            f"{condition.value}"
        )


# ============================================================
# WALK-FORWARD PERIODS
# ============================================================

def create_periods(
    m15
):

    years = sorted(
        m15["timestamp"]
        .dt.year
        .unique()
    )

    periods = []

    # --------------------------------------------------------
    # Use 3 years training -> 1 year testing
    # --------------------------------------------------------

    for i in range(
        len(years) - 3
    ):

        train_years = years[
            i:i + 3
        ]

        test_year = years[
            i + 3
        ]

        train_start = pd.Timestamp(
            f"{train_years[0]}-01-01"
        )

        train_end = pd.Timestamp(
            f"{train_years[-1]}-12-31 23:59:59"
        )

        test_start = pd.Timestamp(
            f"{test_year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{test_year}-12-31 23:59:59"
        )

        periods.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "label": (
                    f"{train_years[0]}-"
                    f"{train_years[-1]}"
                    f" -> "
                    f"{test_year}"
                ),
            }
        )

    return periods


# ============================================================
# RUN ONE WALK-FORWARD PERIOD
# ============================================================

def run_period(
    period,
    direction,
    full_data,
):

    print()
    print("=" * 70)
    print(
        f"{direction} | "
        f"{period['label']}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Split every timeframe
    # --------------------------------------------------------

    train_data = {}
    test_data = {}

    for timeframe, df in full_data.items():

        df = df.sort_values(
            "timestamp"
        )

        train_data[timeframe] = (
            df[
                (df["timestamp"] >= period["train_start"])
                & (df["timestamp"] <= period["train_end"])
            ]
            .copy()
            .reset_index(drop=True)
        )

        test_data[timeframe] = (
            df[
                (df["timestamp"] >= period["test_start"])
                & (df["timestamp"] <= period["test_end"])
            ]
            .copy()
            .reset_index(drop=True)
        )

    train_m15 = train_data[
        "M15"
    ]

    test_m15 = test_data[
        "M15"
    ]

    if train_m15.empty or test_m15.empty:

        print(
            "Skipping period: "
            "insufficient data."
        )

        return None

    print(
        f"Training candles: "
        f"{len(train_m15):,}"
    )

    print(
        f"Testing candles:  "
        f"{len(test_m15):,}"
    )

    # --------------------------------------------------------
    # Optimize training
    # --------------------------------------------------------

    print()
    print(
        f"Optimizing {direction}..."
    )

    training_results = optimize_training(
        direction,
        train_data,
        train_m15,
    )

    if not training_results:

        print(
            "No valid training "
            "strategies found."
        )

        return None

    best = training_results[
        0
    ]

    print()
    print(
        "BEST TRAINING STRATEGY"
    )

    print(
        f"Trades:        "
        f"{best['trades']}"
    )

    print(
        f"Win Rate:      "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{best['profit_factor']:.2f}"
    )

    print(
        f"Net R:         "
        f"{best['net_r']:.2f}"
    )

    print(
        f"Max Drawdown:  "
        f"{best['max_drawdown_r']:.2f}R"
    )

    print_strategy(
        best
    )

    # --------------------------------------------------------
    # OUT-OF-SAMPLE TEST
    # --------------------------------------------------------

    print()
    print(
        "OUT-OF-SAMPLE TEST"
    )

    # IMPORTANT:
    # We use the exact frozen strategy.
    # Nothing from the test period is used
    # to choose it.

    test_result = evaluate_strategy(
        best["conditions"],
        direction,
        test_data,
        test_m15,
    )

    if test_result is None:

        print(
            "Test produced "
            "no trades."
        )

        return {
            "label": period["label"],
            "direction": direction,
            "training": best,
            "testing": None,
        }

    print(
        f"Test Trades:   "
        f"{test_result['trades']}"
    )

    print(
        f"Test Win Rate: "
        f"{test_result['win_rate']:.2f}%"
    )

    print(
        f"Test PF:        "
        f"{test_result['profit_factor']:.2f}"
    )

    print(
        f"Test Net R:     "
        f"{test_result['net_r']:.2f}"
    )

    print(
        f"Test Drawdown:  "
        f"{test_result['max_drawdown_r']:.2f}R"
    )

    return {
        "label": period["label"],
        "direction": direction,
        "training": best,
        "testing": test_result,
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results,
    direction,
):

    print()
    print("=" * 70)
    print(
        f"{direction} WALK-FORWARD SUMMARY"
    )
    print("=" * 70)

    valid = [
        r
        for r in results
        if r["testing"] is not None
    ]

    if not valid:

        print(
            "No valid out-of-sample "
            "results."
        )

        return

    total_trades = sum(
        r["testing"]["trades"]
        for r in valid
    )

    total_net_r = sum(
        r["testing"]["net_r"]
        for r in valid
    )

    total_wins = sum(
        r["testing"]["wins"]
        for r in valid
    )

    total_losses = sum(
        r["testing"]["losses"]
        for r in valid
    )

    gross_profit = sum(
        sum(
            trade.pnl
            for trade in r["testing"]["trades_data"]
            if trade.pnl > 0
        )
        for r in valid
    )

    gross_loss = abs(
        sum(
            sum(
                trade.pnl
                for trade in r["testing"]["trades_data"]
                if trade.pnl < 0
            )
            for r in valid
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 999.0
    )

    win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades > 0
        else 0
    )

    profitable_periods = sum(
        r["testing"]["net_r"] > 0
        for r in valid
    )

    print()

    print(
        "Period                         "
        "Trades     PF      Net R"
    )

    print(
        "-" * 70
    )

    for result in valid:

        test = result[
            "testing"
        ]

        print(
            f"{result['label']:<30}"
            f"{test['trades']:>6}     "
            f"{test['profit_factor']:>5.2f}    "
            f"{test['net_r']:>6.2f}"
        )

    print(
        "-" * 70
    )

    print(
        f"TOTAL OOS TRADES:       "
        f"{total_trades}"
    )

    print(
        f"TOTAL OOS WINS:         "
        f"{total_wins}"
    )

    print(
        f"TOTAL OOS LOSSES:       "
        f"{total_losses}"
    )

    print(
        f"OOS WIN RATE:           "
        f"{win_rate:.2f}%"
    )

    print(
        f"OOS PROFIT FACTOR:      "
        f"{profit_factor:.2f}"
    )

    print(
        f"OOS TOTAL NET R:        "
        f"{total_net_r:.2f}"
    )

    print(
        f"PROFITABLE PERIODS:     "
        f"{profitable_periods}/"
        f"{len(valid)}"
    )

    print()

    if (
        total_net_r > 0
        and profitable_periods
        >= len(valid) * 0.6
        and total_trades >= 20
    ):

        print(
            "STATUS: PROMISING"
        )

    else:

        print(
            "STATUS: NOT YET ROBUST"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NEXA FUNDS AI"
    )
    print(
        "MULTI-PERIOD WALK-FORWARD "
        "VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "Loading indicators..."
    )

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

    m15 = (
        data["M15"]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    m15["timestamp"] = pd.to_datetime(
        m15["timestamp"]
    )

    for timeframe in data:

        data[timeframe][
            "timestamp"
        ] = pd.to_datetime(
            data[timeframe]["timestamp"]
        )

    print(
        f"M15 candles: "
        f"{len(m15):,}"
    )

    # --------------------------------------------------------
    # PERIODS
    # --------------------------------------------------------

    periods = create_periods(
        m15
    )

    print()
    print("=" * 70)
    print(
        "WALK-FORWARD PERIODS"
    )
    print("=" * 70)

    for period in periods:

        print(
            period["label"]
        )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUY")
    print("=" * 70)

    buy_results = []

    for period in periods:

        result = run_period(
            period,
            "BUY",
            data,
        )

        if result is not None:

            buy_results.append(
                result
            )

    summarize(
        buy_results,
        "BUY",
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELL")
    print("=" * 70)

    sell_results = []

    for period in periods:

        result = run_period(
            period,
            "SELL",
            data,
        )

        if result is not None:

            sell_results.append(
                result
            )

    summarize(
        sell_results,
        "SELL",
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MULTI-PERIOD WALK-FORWARD "
        "VALIDATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()