from __future__ import annotations

import pandas as pd

from app.forecasting.uncertainty.error_profiles import (
    calculate_error_profiles,
)

from app.forecasting.uncertainty.prediction_uncertainty import (
    EmpiricalPredictionInterval,
)

from app.forecasting.uncertainty.reliability import (
    calculate_interval_reliability,
)

from app.forecasting.uncertainty.confidence import (
    calculate_confidence,
)


class UncertaintyService:

    def __init__(
        self,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ):

        self.interval_model = (
            EmpiricalPredictionInterval(
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
        )

        self.error_profile = None

    def fit(
        self,
        actual: pd.Series,
        prediction: pd.Series,
    ) -> dict:

        self.error_profile = (
            calculate_error_profiles(
                actual,
                prediction,
            )
        )

        self.interval_model.fit(
            actual,
            prediction,
        )

        return self.error_profile

    def predict(
        self,
        predictions,
    ) -> pd.DataFrame:

        result = (
            self.interval_model
            .predict_interval(
                predictions
            )
        )

        result["confidence"] = (
            calculate_confidence(
                result["prediction"],
                result["lower_bound"],
                result["upper_bound"],
            )
        )

        return result

    def evaluate(
        self,
        actual: pd.Series,
        prediction_df: pd.DataFrame,
    ) -> dict:

        return calculate_interval_reliability(
            actual,
            prediction_df[
                "lower_bound"
            ],
            prediction_df[
                "upper_bound"
            ],
        )