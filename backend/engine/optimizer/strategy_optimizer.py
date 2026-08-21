from dataclasses import dataclass, field
from typing import Any, Dict, List
import itertools
import pandas as pd


@dataclass
class Parameter:
    name: str
    values: List[Any]


@dataclass
class OptimizationResult:
    parameters: Dict[str, Any]
    metrics: Dict[str, float]


class StrategyOptimizer:
    """
    NEXA FUNDS AI
    Strategy parameter optimization engine.

    Generates parameter combinations, evaluates them using
    a supplied backtest function and ranks the results.
    """

    def __init__(self):
        self.results: List[OptimizationResult] = []

    def generate_combinations(
        self,
        parameters: List[Parameter]
    ) -> List[Dict[str, Any]]:

        if not parameters:
            return [{}]

        names = [p.name for p in parameters]
        values = [p.values for p in parameters]

        combinations = []

        for combination in itertools.product(*values):
            combinations.append(
                dict(zip(names, combination))
            )

        return combinations

    def optimize(
        self,
        data: pd.DataFrame,
        parameters: List[Parameter],
        strategy_function,
        backtest_function
    ):

        combinations = self.generate_combinations(parameters)

        self.results = []

        for params in combinations:

            strategy_data = strategy_function(
                data.copy(),
                params
            )

            trades = backtest_function(
                strategy_data,
                params
            )

            metrics = self.calculate_metrics(trades)

            self.results.append(
                OptimizationResult(
                    parameters=params,
                    metrics=metrics
                )
            )

        return self.rank_results()

    def calculate_metrics(self, trades):

        if trades is None or len(trades) == 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_losing_streak": 0,
                "max_winning_streak": 0
            }

        pnl = pd.Series(
            [float(t) for t in trades]
        )

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        win_rate = (
            len(wins) / len(pnl) * 100
            if len(pnl) > 0
            else 0.0
        )

        expectancy = pnl.mean()

        max_win = 0
        max_loss = 0

        current_win = 0
        current_loss = 0

        for value in pnl:

            if value > 0:
                current_win += 1
                current_loss = 0
            elif value < 0:
                current_loss += 1
                current_win = 0
            else:
                current_win = 0
                current_loss = 0

            max_win = max(max_win, current_win)
            max_loss = max(max_loss, current_loss)

        return {
            "trades": int(len(pnl)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": float(win_rate),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "net_profit": float(pnl.sum()),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "max_losing_streak": int(max_loss),
            "max_winning_streak": int(max_win)
        }

    def rank_results(self):

        if not self.results:
            return pd.DataFrame()

        rows = []

        for result in self.results:

            row = {
                **result.parameters,
                **result.metrics
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Composite score.
        # This deliberately rewards profitability and consistency
        # instead of simply selecting the highest raw profit.

        df["optimization_score"] = (
            df["net_profit"]
            + (df["expectancy"] * 10)
            + (df["profit_factor"].replace(
                [float("inf")],
                100
            ) * 20)
            + (df["win_rate"] * 2)
            - (df["max_losing_streak"] * 15)
        )

        df = df.sort_values(
            "optimization_score",
            ascending=False
        ).reset_index(drop=True)

        return df

    def best(self):

        ranked = self.rank_results()

        if ranked.empty:
            return None

        return ranked.iloc[0].to_dict()
