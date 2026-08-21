from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Feature(ABC):

    name: str = "feature"

    @abstractmethod
    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> pd.Series | pd.DataFrame:

        raise NotImplementedError


class FeatureEngine:

    def __init__(self):

        self.features: dict[str, Feature] = {}

    def register(self, feature: Feature):

        feature_name = feature.name.lower()

        if feature_name in self.features:
            raise ValueError(
                f"Feature '{feature_name}' is already registered."
            )

        self.features[feature_name] = feature

    def calculate(
        self,
        name: str,
        data: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ):

        feature_name = name.lower()

        if feature_name not in self.features:

            raise ValueError(
                f"Feature '{name}' is not registered."
            )

        return self.features[feature_name].calculate(
            data,
            parameters or {},
        )

    def list_features(self):

        return sorted(self.features.keys())

    def calculate_all(
        self,
        data: pd.DataFrame,
        requests: list[dict[str, Any]],
    ) -> pd.DataFrame:

        result = data.copy()

        for request in requests:

            feature_name = request["name"]

            parameters = request.get(
                "parameters",
                {},
            )

            values = self.calculate(
                feature_name,
                result,
                parameters,
            )

            if isinstance(values, pd.Series):

                column_name = request.get(
                    "output",
                    feature_name,
                )

                result[column_name] = values

            elif isinstance(values, pd.DataFrame):

                result = pd.concat(
                    [
                        result,
                        values,
                    ],
                    axis=1,
                )

            else:

                raise TypeError(
                    f"Feature '{feature_name}' returned "
                    f"an unsupported type."
                )

        return result
