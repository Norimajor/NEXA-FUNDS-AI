import pandas as pd
import numpy as np


class RegimePerformanceEngine:

    def __init__(self):

        pass


    def _metrics(self, trades):

        if len(trades) == 0:

            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
            }


        pnl = pd.to_numeric(
            trades["pnl"],
            errors="coerce"
        ).fillna(0)


        results = (
            trades["result"]
            .astype(str)
            .str.upper()
        )


        total = len(trades)

        wins = int(
            (results == "WIN").sum()
        )

        losses = int(
            (results == "LOSS").sum()
        )


        gross_profit = pnl[
            pnl > 0
        ].sum()


        gross_loss = abs(
            pnl[
                pnl < 0
            ].sum()
        )


        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = (
                np.inf
                if gross_profit > 0
                else 0.0
            )


        return {

            "trades": total,

            "wins": wins,

            "losses": losses,

            "win_rate": (
                wins / total * 100
            ),

            "net_profit": pnl.sum(),

            "gross_profit":
                gross_profit,

            "gross_loss":
                gross_loss,

            "profit_factor":
                profit_factor,

            "expectancy":
                pnl.mean(),

        }


    def analyze(
        self,
        trades,
        regime_column
    ):

        if trades is None:

            return {}


        if len(trades) == 0:

            return {}


        if regime_column not in trades.columns:

            raise ValueError(
                f"Missing regime column: "
                f"{regime_column}"
            )


        results = {}


        for regime, group in trades.groupby(
            regime_column,
            dropna=False
        ):

            if pd.isna(regime):

                regime_name = "UNKNOWN"

            else:

                regime_name = str(
                    regime
                )


            results[
                regime_name
            ] = self._metrics(
                group
            )


        return results


    def analyze_multiple(
        self,
        trades,
        columns
    ):

        output = {}


        for column in columns:

            output[column] = (
                self.analyze(
                    trades,
                    column
                )
            )


        return output


    def rank_regimes(
        self,
        analysis,
        metric="profit_factor"
    ):

        rows = []


        for regime, metrics in (
            analysis.items()
        ):

            rows.append({

                "regime": regime,

                metric:
                    metrics.get(
                        metric,
                        0
                    ),

                "trades":
                    metrics.get(
                        "trades",
                        0
                    ),

                "win_rate":
                    metrics.get(
                        "win_rate",
                        0
                    ),

                "expectancy":
                    metrics.get(
                        "expectancy",
                        0
                    ),

            })


        if not rows:

            return pd.DataFrame()


        result = pd.DataFrame(
            rows
        )


        return result.sort_values(
            metric,
            ascending=False
        ).reset_index(
            drop=True
        )
