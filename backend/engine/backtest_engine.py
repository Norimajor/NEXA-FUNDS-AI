import pandas as pd


class BacktestEngine:

    def __init__(
        self,
        initial_balance=10000.0,
        risk_percent=1.0,
        reward_risk=2.0,
        spread=0.0,
        commission=0.0,
    ):

        self.initial_balance = float(initial_balance)
        self.risk_percent = float(risk_percent)
        self.reward_risk = float(reward_risk)
        self.spread = float(spread)
        self.commission = float(commission)

    def validate_data(self, data):

        required = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "signal",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def calculate_position_size(
        self,
        balance,
        entry,
        stop_loss,
    ):

        risk_amount = (
            balance
            * self.risk_percent
            / 100.0
        )

        stop_distance = abs(
            entry - stop_loss
        )

        if stop_distance <= 0:
            return 0.0

        return (
            risk_amount
            /
            stop_distance
        )

    def run(
        self,
        data,
        stop_distance,
    ):

        self.validate_data(data)

        if stop_distance <= 0:
            raise ValueError(
                "stop_distance must be greater than zero."
            )

        data = data.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        balance = self.initial_balance
        trades = []
        position = None

        for i in range(len(data)):

            row = data.iloc[i]

            if position is not None:

                exit_price = None
                exit_reason = None

                if position["direction"] == "BUY":

                    sl_hit = (
                        row["low"]
                        <=
                        position["stop_loss"]
                    )

                    tp_hit = (
                        row["high"]
                        >=
                        position["take_profit"]
                    )

                    if sl_hit and tp_hit:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL_AND_TP_SAME_CANDLE"

                    elif sl_hit:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL"

                    elif tp_hit:
                        exit_price = position["take_profit"]
                        exit_reason = "TP"

                else:

                    sl_hit = (
                        row["high"]
                        >=
                        position["stop_loss"]
                    )

                    tp_hit = (
                        row["low"]
                        <=
                        position["take_profit"]
                    )

                    if sl_hit and tp_hit:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL_AND_TP_SAME_CANDLE"

                    elif sl_hit:
                        exit_price = position["stop_loss"]
                        exit_reason = "SL"

                    elif tp_hit:
                        exit_price = position["take_profit"]
                        exit_reason = "TP"

                if exit_price is not None:

                    if position["direction"] == "BUY":
                        price_change = (
                            exit_price
                            -
                            position["entry"]
                        )
                    else:
                        price_change = (
                            position["entry"]
                            -
                            exit_price
                        )

                    pnl = (
                        price_change
                        *
                        position["size"]
                    )

                    pnl -= self.commission
                    balance += pnl

                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": row["timestamp"],
                        "direction": position["direction"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "stop_loss": position["stop_loss"],
                        "take_profit": position["take_profit"],
                        "size": position["size"],
                        "pnl": pnl,
                        "balance": balance,
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "exit_reason": exit_reason,
                    })

                    position = None
                    continue

            signal = str(
                row["signal"]
            ).upper()

            if signal not in {"BUY", "SELL"}:
                continue

            entry = float(row["close"])

            if signal == "BUY":

                entry += self.spread / 2

                stop_loss = (
                    entry
                    -
                    stop_distance
                )

                take_profit = (
                    entry
                    +
                    stop_distance
                    *
                    self.reward_risk
                )

            else:

                entry -= self.spread / 2

                stop_loss = (
                    entry
                    +
                    stop_distance
                )

                take_profit = (
                    entry
                    -
                    stop_distance
                    *
                    self.reward_risk
                )

            size = self.calculate_position_size(
                balance,
                entry,
                stop_loss,
            )

            if size <= 0:
                continue

            position = {
                "entry_time": row["timestamp"],
                "direction": signal,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": size,
            }

        if position is not None:

            last_row = data.iloc[-1]

            exit_price = float(
                last_row["close"]
            )

            if position["direction"] == "BUY":

                price_change = (
                    exit_price
                    -
                    position["entry"]
                )

            else:

                price_change = (
                    position["entry"]
                    -
                    exit_price
                )

            pnl = (
                price_change
                *
                position["size"]
            )

            pnl -= self.commission
            balance += pnl

            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": last_row["timestamp"],
                "direction": position["direction"],
                "entry": position["entry"],
                "exit": exit_price,
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "size": position["size"],
                "pnl": pnl,
                "balance": balance,
                "result": "WIN" if pnl > 0 else "LOSS",
                "exit_reason": "END_OF_DATA",
            })

        return {
            "initial_balance": self.initial_balance,
            "final_balance": balance,
            "net_profit": balance - self.initial_balance,
            "trades": pd.DataFrame(trades),
        }
