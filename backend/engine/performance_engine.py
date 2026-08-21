import numpy as np
import pandas as pd


class PerformanceEngine:

    def __init__(self, initial_balance=None):
        self.initial_balance = initial_balance

    def calculate_streaks(self, results):

        max_win = 0
        max_loss = 0

        current_win = 0
        current_loss = 0

        current_streak = 0
        current_type = "NONE"

        for result in results:

            if result == "WIN":

                current_win += 1
                current_loss = 0

                max_win = max(
                    max_win,
                    current_win
                )

                current_streak = current_win
                current_type = "WIN"

            elif result == "LOSS":

                current_loss += 1
                current_win = 0

                max_loss = max(
                    max_loss,
                    current_loss
                )

                current_streak = current_loss
                current_type = "LOSS"

        return {
            "max_winning_streak": max_win,
            "max_losing_streak": max_loss,
            "current_streak": current_streak,
            "current_streak_type": current_type,
        }

    def calculate_drawdown(self, balances):

        if len(balances) == 0:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_percent": 0.0,
            }

        balances = pd.Series(
            balances,
            dtype=float
        )

        peak = balances.cummax()

        drawdown = balances - peak

        drawdown_percent = (
            drawdown
            /
            peak.replace(0, np.nan)
        ) * 100

        return {
            "max_drawdown":
                abs(float(drawdown.min())),

            "max_drawdown_percent":
                abs(float(drawdown_percent.min())),
        }

    def calculate(self, trades):

        if trades is None:
            return self.empty_result()

        if isinstance(trades, dict):
            trades = trades.get(
                "trades",
                pd.DataFrame()
            )

        if len(trades) == 0:
            return self.empty_result()

        trades = trades.copy()

        pnl = pd.to_numeric(
            trades["pnl"],
            errors="coerce"
        ).fillna(0)

        results = (
            trades["result"]
            .astype(str)
            .str.upper()
        )

        total_trades = len(trades)

        wins = int(
            (results == "WIN").sum()
        )

        losses = int(
            (results == "LOSS").sum()
        )

        win_rate = (
            wins
            /
            total_trades
            *
            100
        )

        loss_rate = (
            losses
            /
            total_trades
            *
            100
        )

        gross_profit = float(
            pnl[pnl > 0].sum()
        )

        gross_loss = float(
            abs(pnl[pnl < 0].sum())
        )

        net_profit = float(
            pnl.sum()
        )

        average_trade = float(
            pnl.mean()
        )

        average_winner = (
            float(pnl[pnl > 0].mean())
            if wins > 0
            else 0.0
        )

        average_loser = (
            float(pnl[pnl < 0].mean())
            if losses > 0
            else 0.0
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = (
                float("inf")
                if gross_profit > 0
                else 0.0
            )

        expectancy = average_trade

        if self.initial_balance is not None:

            balances = pd.concat(
                [
                    pd.Series([
                        self.initial_balance
                    ]),
                    trades["balance"],
                ],
                ignore_index=True,
            )

        else:

            balances = trades["balance"]

        drawdown = self.calculate_drawdown(
            balances
        )

        streaks = self.calculate_streaks(
            results.tolist()
        )

        previous_balances = balances.iloc[:-1]

        returns = (
            pnl.reset_index(drop=True)
            /
            previous_balances.reset_index(
                drop=True
            )
        )

        returns = (
            pd.Series(returns)
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(returns) > 1:

            std = returns.std(ddof=1)

            sharpe = (
                (
                    returns.mean()
                    /
                    std
                )
                *
                np.sqrt(len(returns))
                if std > 0
                else 0.0
            )

            downside = returns[
                returns < 0
            ]

            if len(downside) > 1:

                downside_std = downside.std(
                    ddof=1
                )

                sortino = (
                    (
                        returns.mean()
                        /
                        downside_std
                    )
                    *
                    np.sqrt(len(returns))
                    if downside_std > 0
                    else 0.0
                )

            else:

                sortino = 0.0

        else:

            sharpe = 0.0
            sortino = 0.0

        if drawdown["max_drawdown"] > 0:

            recovery_factor = (
                net_profit
                /
                drawdown["max_drawdown"]
            )

        else:

            recovery_factor = (
                float("inf")
                if net_profit > 0
                else 0.0
            )

        return {

            "total_trades": total_trades,
            "winning_trades": wins,
            "losing_trades": losses,

            "win_rate": win_rate,
            "loss_rate": loss_rate,

            "gross_profit": gross_profit,
            "gross_loss": gross_loss,

            "net_profit": net_profit,

            "average_trade": average_trade,
            "average_winner": average_winner,
            "average_loser": average_loser,

            "profit_factor": profit_factor,
            "expectancy": expectancy,

            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),

            "recovery_factor":
                float(recovery_factor),

            "max_drawdown":
                drawdown["max_drawdown"],

            "max_drawdown_percent":
                drawdown["max_drawdown_percent"],

            **streaks,

            "best_trade":
                float(pnl.max()),

            "worst_trade":
                float(pnl.min()),
        }

    def empty_result(self):

        return {

            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,

            "win_rate": 0.0,
            "loss_rate": 0.0,

            "gross_profit": 0.0,
            "gross_loss": 0.0,

            "net_profit": 0.0,

            "average_trade": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,

            "profit_factor": 0.0,
            "expectancy": 0.0,

            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,

            "recovery_factor": 0.0,

            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,

            "max_winning_streak": 0,
            "max_losing_streak": 0,

            "current_streak": 0,
            "current_streak_type": "NONE",

            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
