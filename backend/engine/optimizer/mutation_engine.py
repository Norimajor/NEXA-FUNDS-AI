from dataclasses import dataclass
from typing import Dict, List, Any
import copy


@dataclass
class Mutation:
    name: str
    category: str
    parameters: Dict[str, Any]


class StrategyMutationEngine:
    """
    NEXA FUNDS AI
    Automatically generates controlled strategy modifications.

    The engine does NOT decide whether a mutation is good.
    It generates candidates for the backtesting/optimization layer.
    """

    def __init__(self):

        self.mutation_library = {

            "trend": [

                Mutation(
                    name="ADD_EMA_FILTER",
                    category="trend",
                    parameters={
                        "indicator": "EMA",
                        "periods": [20, 50]
                    }
                ),

                Mutation(
                    name="ADD_EMA_FILTER_50_200",
                    category="trend",
                    parameters={
                        "indicator": "EMA",
                        "periods": [50, 200]
                    }
                ),

                Mutation(
                    name="ADD_ADX_FILTER",
                    category="trend",
                    parameters={
                        "indicator": "ADX",
                        "period": 14,
                        "threshold": 20
                    }
                ),

                Mutation(
                    name="ADD_HMA_FILTER",
                    category="trend",
                    parameters={
                        "indicator": "HMA",
                        "period": 50
                    }
                )
            ],

            "momentum": [

                Mutation(
                    name="ADD_RSI_FILTER",
                    category="momentum",
                    parameters={
                        "indicator": "RSI",
                        "period": 14,
                        "long_threshold": 50,
                        "short_threshold": 50
                    }
                ),

                Mutation(
                    name="ADD_CCI_FILTER",
                    category="momentum",
                    parameters={
                        "indicator": "CCI",
                        "period": 20
                    }
                ),

                Mutation(
                    name="ADD_STOCHASTIC_FILTER",
                    category="momentum",
                    parameters={
                        "indicator": "STOCHASTIC",
                        "period": 14
                    }
                ),

                Mutation(
                    name="ADD_ROC_FILTER",
                    category="momentum",
                    parameters={
                        "indicator": "ROC",
                        "period": 10
                    }
                )
            ],

            "volatility": [

                Mutation(
                    name="ADD_ATR_FILTER",
                    category="volatility",
                    parameters={
                        "indicator": "ATR",
                        "period": 14
                    }
                ),

                Mutation(
                    name="ADD_BOLLINGER_FILTER",
                    category="volatility",
                    parameters={
                        "indicator": "BOLLINGER_BANDS",
                        "period": 20
                    }
                ),

                Mutation(
                    name="ADD_VOLATILITY_REGIME",
                    category="volatility",
                    parameters={
                        "feature": "VOLATILITY_REGIME"
                    }
                )
            ],

            "volume": [

                Mutation(
                    name="ADD_VWAP_FILTER",
                    category="volume",
                    parameters={
                        "indicator": "VWAP"
                    }
                ),

                Mutation(
                    name="ADD_MFI_FILTER",
                    category="volume",
                    parameters={
                        "indicator": "MFI",
                        "period": 14
                    }
                ),

                Mutation(
                    name="ADD_RELATIVE_VOLUME",
                    category="volume",
                    parameters={
                        "indicator": "RELATIVE_VOLUME",
                        "period": 20
                    }
                )
            ],

            "price_action": [

                Mutation(
                    name="ADD_PIN_BAR",
                    category="price_action",
                    parameters={
                        "pattern": "PIN_BAR"
                    }
                ),

                Mutation(
                    name="ADD_ENGULFING",
                    category="price_action",
                    parameters={
                        "pattern": "ENGULFING"
                    }
                ),

                Mutation(
                    name="ADD_INSIDE_BAR",
                    category="price_action",
                    parameters={
                        "pattern": "INSIDE_BAR"
                    }
                )
            ],

            "smc": [

                Mutation(
                    name="ADD_BOS",
                    category="smc",
                    parameters={
                        "feature": "BOS"
                    }
                ),

                Mutation(
                    name="ADD_CHOCH",
                    category="smc",
                    parameters={
                        "feature": "CHOCH"
                    }
                ),

                Mutation(
                    name="ADD_FVG",
                    category="smc",
                    parameters={
                        "feature": "FVG"
                    }
                ),

                Mutation(
                    name="ADD_ORDER_BLOCK",
                    category="smc",
                    parameters={
                        "feature": "ORDER_BLOCK"
                    }
                ),

                Mutation(
                    name="ADD_LIQUIDITY_SWEEP",
                    category="smc",
                    parameters={
                        "feature": "LIQUIDITY_SWEEP"
                    }
                ),

                Mutation(
                    name="ADD_PREMIUM_DISCOUNT",
                    category="smc",
                    parameters={
                        "feature": "PREMIUM_DISCOUNT"
                    }
                )
            ]
        }

    def get_all_mutations(self):

        mutations = []

        for category in self.mutation_library:

            mutations.extend(
                self.mutation_library[category]
            )

        return mutations

    def get_mutations_by_category(self, category):

        return self.mutation_library.get(
            category,
            []
        )

    def mutate(self, strategy):

        candidates = []

        base_strategy = copy.deepcopy(strategy)

        # Single mutations

        for mutation in self.get_all_mutations():

            candidate = copy.deepcopy(base_strategy)

            candidate.setdefault(
                "mutations",
                []
            )

            candidate["mutations"].append(
                {
                    "name": mutation.name,
                    "category": mutation.category,
                    "parameters": mutation.parameters
                }
            )

            candidates.append(candidate)

        return candidates

    def combine_mutations(
        self,
        strategy,
        max_mutations=2
    ):

        candidates = []

        mutations = self.get_all_mutations()

        base_strategy = copy.deepcopy(strategy)

        # Controlled combinations

        for i in range(len(mutations)):

            for j in range(
                i + 1,
                len(mutations)
            ):

                selected = [
                    mutations[i],
                    mutations[j]
                ]

                candidate = copy.deepcopy(
                    base_strategy
                )

                candidate["mutations"] = [

                    {
                        "name": m.name,
                        "category": m.category,
                        "parameters": m.parameters
                    }

                    for m in selected

                ]

                candidates.append(candidate)

                if len(candidates) >= 100:

                    return candidates

        return candidates
