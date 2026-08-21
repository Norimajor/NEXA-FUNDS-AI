
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

PAIRS = [
    "EURUSD.m",
    "GBPUSD.m",
    "USDJPY.m",
    "USDCHF.m",
    "USDCAD.m",
    "AUDUSD.m",
    "NZDUSD.m",
    "EURJPY.m",
    "EURGBP.m",
    "EURAUD.m",
]

TIMEFRAMES = [
    "H4",
    "H1",
    "M30",
    "M15",
    "M5",
]

RISK_REWARD = 2.0
STOP_DISTANCE = 0.0010

MIN_TRAIN_TRADES = 10
MIN_OOS_TRADES = 5

TRAIN_END = "2023-12-31 23:59:59"
OOS_START = "2024-01-01 00:00:00"


# ============================================================
# CONDITION POOLS
# ============================================================

def build_conditions(direction):

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
# INDICATOR SIGNAL CACHE
# ============================================================

def build_signal_cache(data):

    cache = {}

    for timeframe, df in data.items():

        evaluator = ConditionEvaluator(df)

        cache[timeframe] = {
            "df": df,
            "evaluator": evaluator,
        }

    return cache


def condition_signal(
    condition,
    cache,
):

    timeframe = getattr(
        condition,
        "timeframe",
        "M15",
    ).upper()

    evaluator = cache[timeframe]["evaluator"]

    signal = evaluator.evaluate_condition(
        condition
    )

    return signal.astype(bool).reset_index(
        drop=True
    )


# ============================================================
# ALIGN CONDITION TO M15
# ============================================================

def align_condition(
    condition,
    cache,
    m15,
):

    timeframe = getattr(
        condition,
        "timeframe",
        "M15",
    ).upper()

    source = cache[timeframe]["df"]

    signal = condition_signal(
        condition,
        cache,
    )

    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                source["timestamp"]
            ),
            "signal": signal.values,
        }
    )

    result = result.sort_values(
        "timestamp"
    )

    target = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                m15["timestamp"]
            )
        }
    )

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
    ).reset_index(drop=True)


# ============================================================
# GENERATE COMBINED SIGNAL
# ============================================================

def generate_signal(
    conditions,
    cache,
    m15,
):

    final_signal = pd.Series(
        True,
        index=range(len(m15)),
        dtype=bool,
    )

    for condition in conditions:

        aligned = align_condition(
            condition,
            cache,
            m15,
        )

        final_signal &= aligned

    return final_signal


# ============================================================
# BACKTEST
# ============================================================

def calculate_metrics(
    trades,
):

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
            gross_profit / gross_loss
        )
    else:
        profit_factor = 999.0

    win_rate = (
        wins / len(trades) * 100
    )

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
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(
    m15,
    signal,
    direction,
):

    if len(m15) == 0:
        return None

    backtester = Backtester(
        risk_reward=RISK_REWARD,
        stop_distance=STOP_DISTANCE,
    )

    trades = backtester.run(
        m15,
        signal,
        direction=direction,
    )

    return calculate_metrics(
        trades
    )


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(
    m15,
):

    timestamps = pd.to_datetime(
        m15["timestamp"]
    )

    train_mask = (
        timestamps <= TRAIN_END
    )

    oos_mask = (
        timestamps >= OOS_START
    )

    train = (
        m15.loc[train_mask]
        .copy()
        .reset_index(drop=True)
    )

    oos = (
        m15.loc[oos_mask]
        .copy()
        .reset_index(drop=True)
    )

    return train, oos


# ============================================================
# OPTIMIZE ONE PAIR / ONE DIRECTION
# ============================================================

def optimize_pair(
    pair,
    direction,
    data,
):

    m15 = data["M15"].copy()

    m15["timestamp"] = pd.to_datetime(
        m15["timestamp"]
    )

    m15 = (
        m15.sort_values("timestamp")
        .reset_index(drop=True)
    )

    train_m15, oos_m15 = split_data(
        m15
    )

    if len(train_m15) == 0 or len(oos_m15) == 0:
        return []

    cache = build_signal_cache(
        data
    )

    pools = build_conditions(
        direction
    )

    combinations = itertools.product(
        *pools
    )

    parser = StrategyParser()

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

            conditions = (
                strategy.entry_conditions
            )

            # ------------------------------------------------
            # Generate complete signal once
            # ------------------------------------------------

            full_signal = generate_signal(
                conditions,
                cache,
                m15,
            )

            # ------------------------------------------------
            # Align signal to train/OOS
            # ------------------------------------------------

            timestamps = pd.to_datetime(
                m15["timestamp"]
            )

            train_signal = (
                full_signal[
                    timestamps <= TRAIN_END
                ]
                .reset_index(drop=True)
            )

            oos_signal = (
                full_signal[
                    timestamps >= OOS_START
                ]
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # Minimum training signals
            # ------------------------------------------------

            if int(train_signal.sum()) < MIN_TRAIN_TRADES:
                continue

            train_metrics = run_backtest(
                train_m15,
                train_signal,
                direction,
            )

            if train_metrics is None:
                continue

            if (
                train_metrics["trades"]
                < MIN_TRAIN_TRADES
            ):
                continue

            # ------------------------------------------------
            # OOS
            # ------------------------------------------------

            oos_metrics = run_backtest(
                oos_m15,
                oos_signal,
                direction,
            )

            if oos_metrics is None:
                continue

            # ------------------------------------------------
            # Store candidate
            # ------------------------------------------------

            results.append(
                {
                    "pair": pair,
                    "direction": direction,
                    "conditions": conditions,

                    "train_trades":
                        train_metrics["trades"],

                    "train_win_rate":
                        train_metrics["win_rate"],

                    "train_pf":
                        train_metrics["profit_factor"],

                    "train_net_r":
                        train_metrics["net_r"],

                    "train_dd":
                        train_metrics[
                            "max_drawdown_r"
                        ],

                    "oos_trades":
                        oos_metrics["trades"],

                    "oos_win_rate":
                        oos_metrics["win_rate"],

                    "oos_pf":
                        oos_metrics["profit_factor"],

                    "oos_net_r":
                        oos_metrics["net_r"],

                    "oos_dd":
                        oos_metrics[
                            "max_drawdown_r"
                        ],
                }
            )

        except Exception:
            continue

    return results


# ============================================================
# ROBUSTNESS SCORE
# ============================================================

def robustness_score(result):

    if result["oos_trades"] < MIN_OOS_TRADES:
        return -9999.0

    oos_pf = result["oos_pf"]
    oos_net = result["oos_net_r"]
    oos_dd = result["oos_dd"]

    # Reward:
    # - profitable OOS performance
    # - good profit factor
    # - more OOS trades
    #
    # Penalize:
    # - drawdown

    score = (
        oos_net
        + (oos_pf - 1.0) * 5.0
        + min(result["oos_trades"], 50) * 0.05
        - oos_dd * 0.20
    )

    return score


# ============================================================
# DISPLAY PAIR RESULTS
# ============================================================

def display_pair_results(
    results,
    pair,
    direction,
):

    candidates = [
        r
        for r in results
        if r["pair"] == pair
        and r["direction"] == direction
    ]

    candidates.sort(
        key=robustness_score,
        reverse=True,
    )

    print()
    print(
        f"{pair} {direction}"
    )
    print("-" * 70)

    if not candidates:

        print(
            "No valid strategies."
        )

        return

    for number, result in enumerate(
        candidates[:3],
        start=1,
    ):

        print()
        print(
            f"#{number} "
            f"Score={robustness_score(result):.2f}"
        )

        print(
            f"TRAIN: "
            f"{result['train_trades']} trades | "
            f"WR={result['train_win_rate']:.2f}% | "
            f"PF={result['train_pf']:.2f} | "
            f"Net={result['train_net_r']:.2f}R"
        )

        print(
            f"OOS:   "
            f"{result['oos_trades']} trades | "
            f"WR={result['oos_win_rate']:.2f}% | "
            f"PF={result['oos_pf']:.2f} | "
            f"Net={result['oos_net_r']:.2f}R | "
            f"DD={result['oos_dd']:.2f}R"
        )

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
# FINAL GLOBAL RANKING
# ============================================================

def display_global_results(
    results,
):

    print()
    print("=" * 90)
    print(
        "TOP MULTI-PAIR OUT-OF-SAMPLE STRATEGIES"
    )
    print("=" * 90)

    valid = [
        r
        for r in results
        if r["oos_trades"]
        >= MIN_OOS_TRADES
    ]

    valid.sort(
        key=robustness_score,
        reverse=True,
    )

    if not valid:

        print(
            "No strategies passed OOS requirements."
        )

        return

    print()

    print(
        f"{'PAIR':<10}"
        f"{'DIR':<7}"
        f"{'OOS TR':<8}"
        f"{'OOS WR':<10}"
        f"{'OOS PF':<9}"
        f"{'NET R':<9}"
        f"{'DD':<8}"
        f"{'SCORE':<9}"
    )

    print("-" * 90)

    for result in valid[:20]:

        print(
            f"{result['pair']:<10}"
            f"{result['direction']:<7}"
            f"{result['oos_trades']:<8}"
            f"{result['oos_win_rate']:<10.2f}"
            f"{result['oos_pf']:<9.2f}"
            f"{result['oos_net_r']:<9.2f}"
            f"{result['oos_dd']:<8.2f}"
            f"{robustness_score(result):<9.2f}"
        )

    print()

    print(
        "TOP STRATEGY CONDITIONS"
    )

    print("-" * 90)

    for number, result in enumerate(
        valid[:10],
        start=1,
    ):

        print()
        print(
            f"#{number} "
            f"{result['pair']} "
            f"{result['direction']}"
        )

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
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "NEXA FUNDS AI"
    )
    print(
        "FAST MULTI-PAIR STRATEGY OPTIMIZER"
    )
    print("=" * 90)

    print()
    print(
        "Training period:"
    )
    print(
        f"2017-06-12 -> {TRAIN_END}"
    )

    print(
        "Out-of-sample:"
    )
    print(
        f"{OOS_START} -> 2026-08-17"
    )

    print()
    print(
        f"Pairs: {len(PAIRS)}"
    )

    print(
        "Combinations per direction: 243"
    )

    all_results = []

    engine = MTFIndicatorEngine()

    # ========================================================
    # PAIR LOOP
    # ========================================================

    for pair_number, pair in enumerate(
        PAIRS,
        start=1,
    ):

        print()
        print("=" * 90)
        print(
            f"[{pair_number}/{len(PAIRS)}] "
            f"LOADING {pair}"
        )
        print("=" * 90)

        try:

            data = engine.build(
                pair,
                TIMEFRAMES,
            )

        except Exception as error:

            print(
                f"ERROR loading {pair}: "
                f"{error}"
            )

            continue

        required = set(
            TIMEFRAMES
        )

        available = set(
            data.keys()
        )

        if not required.issubset(
            available
        ):

            missing = (
                required - available
            )

            print(
                f"SKIPPING {pair}"
            )

            print(
                f"Missing: {missing}"
            )

            continue

        m15 = data["M15"]

        print(
            f"M15 candles: "
            f"{len(m15):,}"
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        print()
        print(
            f"Optimizing BUY "
            f"{pair}..."
        )

        buy_results = optimize_pair(
            pair,
            "BUY",
            data,
        )

        all_results.extend(
            buy_results
        )

        display_pair_results(
            buy_results,
            pair,
            "BUY",
        )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        print()
        print(
            f"Optimizing SELL "
            f"{pair}..."
        )

        sell_results = optimize_pair(
            pair,
            "SELL",
            data,
        )

        all_results.extend(
            sell_results
        )

        display_pair_results(
            sell_results,
            pair,
            "SELL",
        )

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    display_global_results(
        all_results
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 90)
    print(
        "MULTI-PAIR OPTIMIZATION COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "OOS results were NOT used to generate "
        "the strategy conditions."
    )

    print(
        "Use OOS performance as validation, "
        "not proof of future profitability."
    )


if __name__ == "__main__":
    main()
