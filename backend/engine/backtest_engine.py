import pandas as pd


class BacktestEngine:

    def __init__(
        self,
        initial_balance=10000.0,
        risk_percent=1.0,
        reward_risk=2.0,
        spread=0.0,
        commission=0.0,
        stop_on_zero_balance=True,
    ):

        self.initial_balance = float(initial_balance)
        self.risk_percent = float(risk_percent)
        self.reward_risk = float(reward_risk)
        self.spread = float(spread)
        self.commission = float(commission)
        self.stop_on_zero_balance = bool(stop_on_zero_balance)

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
        symbol="UNKNOWN",
        timeframe="UNKNOWN",
        allow_reentry=True,
        max_simultaneous_positions=1,
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
        trade_id = 0
        entries_enabled = True

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
                        "trade_id": position["trade_id"],
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "entry_time": position["entry_time"],
                        "exit_time": row["timestamp"],
                        "direction": position["direction"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "stop_loss": position["stop_loss"],
                        "take_profit": position["take_profit"],
                        "size": position["size"],
                        "pnl": pnl,
                        "gross_pnl": pnl + self.commission,
                        "commission": self.commission,
                        "spread": self.spread,
                        "slippage": 0.0,
                        "r_multiple": (pnl + self.commission) / (position["size"] * abs(position["entry"] - position["stop_loss"])) if position["size"] else 0.0,
                        "balance": balance,
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "exit_reason": exit_reason,
                    })

                    position = None
                    if self.stop_on_zero_balance and balance <= 0:
                        entries_enabled = False
                    entries_enabled = allow_reentry
                    if self.stop_on_zero_balance and balance <= 0:
                        entries_enabled = False
                    continue

            signal = str(
                row["signal"]
            ).upper()

            if signal not in {"BUY", "SELL"} or position is not None or not entries_enabled or max_simultaneous_positions < 1:
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
                "trade_id": trade_id + 1,
                "entry_time": row["timestamp"],
                "direction": signal,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": size,
            }
            trade_id += 1

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
                "trade_id": position["trade_id"],
                "symbol": symbol,
                "timeframe": timeframe,
                "entry_time": position["entry_time"],
                "exit_time": last_row["timestamp"],
                "direction": position["direction"],
                "entry": position["entry"],
                "exit": exit_price,
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "size": position["size"],
                "pnl": pnl,
                "gross_pnl": pnl + self.commission,
                "commission": self.commission,
                "spread": self.spread,
                "slippage": 0.0,
                "r_multiple": (pnl + self.commission) / (position["size"] * abs(position["entry"] - position["stop_loss"])) if position["size"] else 0.0,
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
