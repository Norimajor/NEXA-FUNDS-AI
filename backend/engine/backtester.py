from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    entry: float
    exit: float
    stop_loss: float
    take_profit: float
    result: str
    pnl: float


class Backtester:

    def __init__(
        self,
        risk_reward: float = 2.0,
        stop_distance: float = 0.0010,
    ):
        self.risk_reward = risk_reward
        self.stop_distance = stop_distance

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        direction: str = "BUY",
    ) -> List[Trade]:

        direction = direction.upper()

        if direction not in {"BUY", "SELL"}:
            raise ValueError(
                "direction must be BUY or SELL"
            )

        trades = []

        for i in range(len(data)):

            if not bool(signals.iloc[i]):
                continue

            row = data.iloc[i]

            entry = float(row["close"])
            entry_time = row["timestamp"]

            # ==================================================
            # BUY
            # ==================================================

            if direction == "BUY":

                stop = (
                    entry -
                    self.stop_distance
                )

                target = (
                    entry +
                    self.stop_distance *
                    self.risk_reward
                )

            # ==================================================
            # SELL
            # ==================================================

            else:

                stop = (
                    entry +
                    self.stop_distance
                )

                target = (
                    entry -
                    self.stop_distance *
                    self.risk_reward
                )

            # ==================================================
            # SEARCH FOR EXIT
            # ==================================================

            for j in range(i + 1, len(data)):

                future = data.iloc[j]

                high = float(future["high"])
                low = float(future["low"])

                # ==============================================
                # BUY EXIT
                # ==============================================

                if direction == "BUY":

                    # Stop hit
                    if low <= stop:

                        trades.append(
                            Trade(
                                entry_time=entry_time,
                                exit_time=future["timestamp"],
                                direction="BUY",
                                entry=entry,
                                exit=stop,
                                stop_loss=stop,
                                take_profit=target,
                                result="LOSS",
                                pnl=-1.0,
                            )
                        )

                        break

                    # Target hit
                    if high >= target:

                        trades.append(
                            Trade(
                                entry_time=entry_time,
                                exit_time=future["timestamp"],
                                direction="BUY",
                                entry=entry,
                                exit=target,
                                stop_loss=stop,
                                take_profit=target,
                                result="WIN",
                                pnl=self.risk_reward,
                            )
                        )

                        break

                # ==============================================
                # SELL EXIT
                # ==============================================

                else:

                    # Stop hit
                    if high >= stop:

                        trades.append(
                            Trade(
                                entry_time=entry_time,
                                exit_time=future["timestamp"],
                                direction="SELL",
                                entry=entry,
                                exit=stop,
                                stop_loss=stop,
                                take_profit=target,
                                result="LOSS",
                                pnl=-1.0,
                            )
                        )

                        break

                    # Target hit
                    if low <= target:

                        trades.append(
                            Trade(
                                entry_time=entry_time,
                                exit_time=future["timestamp"],
                                direction="SELL",
                                entry=entry,
                                exit=target,
                                stop_loss=stop,
                                take_profit=target,
                                result="WIN",
                                pnl=self.risk_reward,
                            )
                        )

                        break

        return trades

    @staticmethod
    def report(trades):

        if not trades:

            print("NO TRADES")
            return

        wins = sum(
            1
            for trade in trades
            if trade.result == "WIN"
        )

        losses = len(trades) - wins

        win_rate = (
            wins / len(trades) * 100
        )

        total_pnl = sum(
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
            else float("inf")
        )

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for trade in trades:

            equity += trade.pnl

            peak = max(
                peak,
                equity
            )

            drawdown = (
                peak - equity
            )

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

        print()
        print("=" * 60)
        print("NEXA FUNDS AI BACKTEST")
        print("=" * 60)

        print(
            f"Trades:        {len(trades):,}"
        )

        print(
            f"Wins:          {wins:,}"
        )

        print(
            f"Losses:        {losses:,}"
        )

        print(
            f"Win Rate:      {win_rate:.2f}%"
        )

        print(
            f"Profit Factor: {profit_factor:.2f}"
        )

        print(
            f"Net R:         {total_pnl:.2f}"
        )

        print(
            f"Max Drawdown:  {max_drawdown:.2f}R"
        )

        print("=" * 60)