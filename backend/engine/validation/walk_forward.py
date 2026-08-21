from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import pandas as pd


@dataclass
class ValidationWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardEngine:

    """
    NEXA FUNDS AI

    Walk-forward validation engine.

    Splits historical data into sequential training
    and unseen testing periods.

    Example:

        TRAIN TRAIN TRAIN TRAIN | TEST TEST
        TRAIN TRAIN TRAIN TRAIN | TEST TEST
        TRAIN TRAIN TRAIN TRAIN | TEST TEST

    This helps detect strategies that only perform
    well on the data used to optimize them.
    """

    def __init__(
        self,
        train_size=0.60,
        test_size=0.20,
        step_size=0.20
    ):

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def generate_windows(
        self,
        data_length: int
    ) -> List[ValidationWindow]:

        windows = []

        if data_length <= 0:
            return windows

        train_length = int(
            data_length * self.train_size
        )

        test_length = int(
            data_length * self.test_size
        )

        step_length = int(
            data_length * self.step_size
        )

        if train_length <= 0 or test_length <= 0:
            return windows

        start = 0

        while True:

            train_start = start
            train_end = start + train_length

            test_start = train_end
            test_end = test_start + test_length

            if test_end > data_length:
                break

            windows.append(
                ValidationWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                )
            )

            start += step_length

        return windows

    def evaluate_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        strategy_function: Callable,
        backtest_function: Callable,
        parameters: Dict[str, Any]
    ):

        # Train / optimize only using training data.

        trained_strategy = strategy_function(
            train_data.copy(),
            parameters
        )

        train_trades = backtest_function(
            trained_strategy,
            parameters
        )

        # Test on completely unseen data.

        test_strategy = strategy_function(
            test_data.copy(),
            parameters
        )

        test_trades = backtest_function(
            test_strategy,
            parameters
        )

        train_metrics = self.calculate_metrics(
            train_trades
        )

        test_metrics = self.calculate_metrics(
            test_trades
        )

        return {
            "train": train_metrics,
            "test": test_metrics
        }

    def calculate_metrics(self, trades):

        if trades is None or len(trades) == 0:

            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0
            }

        pnl = pd.Series(
            [float(x) for x in trades]
        )

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = (
                float("inf")
                if gross_profit > 0
                else 0.0
            )

        return {
            "trades": int(len(pnl)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": float(
                len(wins) / len(pnl) * 100
            ),
            "net_profit": float(
                pnl.sum()
            ),
            "profit_factor": float(
                profit_factor
            ),
            "expectancy": float(
                pnl.mean()
            )
        }

    def run(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        backtest_function: Callable,
        parameters: Dict[str, Any]
    ):

        windows = self.generate_windows(
            len(data)
        )

        results = []

        for index, window in enumerate(
            windows,
            start=1
        ):

            train_data = data.iloc[
                window.train_start:
                window.train_end
            ]

            test_data = data.iloc[
                window.test_start:
                window.test_end
            ]

            evaluation = self.evaluate_window(
                train_data,
                test_data,
                strategy_function,
                backtest_function,
                parameters
            )

            results.append({

                "window": index,

                "train_start": window.train_start,

                "train_end": window.train_end,

                "test_start": window.test_start,

                "test_end": window.test_end,

                "train_net_profit":
                    evaluation["train"]["net_profit"],

                "test_net_profit":
                    evaluation["test"]["net_profit"],

                "train_profit_factor":
                    evaluation["train"]["profit_factor"],

                "test_profit_factor":
                    evaluation["test"]["profit_factor"],

                "train_win_rate":
                    evaluation["train"]["win_rate"],

                "test_win_rate":
                    evaluation["test"]["win_rate"],

                "train_expectancy":
                    evaluation["train"]["expectancy"],

                "test_expectancy":
                    evaluation["test"]["expectancy"]

            })

        return pd.DataFrame(results)

    def robustness_score(
        self,
        results: pd.DataFrame
    ):

        if results.empty:
            return 0.0

        profitable_windows = (
            results["test_net_profit"] > 0
        ).sum()

        total_windows = len(results)

        consistency = (
            profitable_windows /
            total_windows
        )

        positive_expectancy = (
            results["test_expectancy"] > 0
        ).sum() / total_windows

        positive_pf = (
            results["test_profit_factor"] > 1
        ).sum() / total_windows

        score = (
            consistency * 40
            + positive_expectancy * 30
            + positive_pf * 30
        )

        return round(
            float(score),
            2
        )
