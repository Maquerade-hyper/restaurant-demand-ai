from __future__ import annotations

import numpy as np
import pandas as pd


class EmpiricalPredictionInterval:

    def __init__(
        self,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        self.lower_error = None
        self.upper_error = None

    def fit(
        self,
        actual: pd.Series,
        prediction: pd.Series,
    ) -> "EmpiricalPredictionInterval":

        errors = (
            actual.to_numpy()
            - prediction.to_numpy()
        )

        errors = errors[
            np.isfinite(errors)
        ]

        if len(errors) == 0:
            raise ValueError(
                "No valid residuals available."
            )

        self.lower_error = float(
            np.quantile(
                errors,
                self.lower_quantile,
            )
        )

        self.upper_error = float(
            np.quantile(
                errors,
                self.upper_quantile,
            )
        )

        return self

    def predict_interval(
        self,
        predictions,
    ) -> pd.DataFrame:

        if (
            self.lower_error is None
            or self.upper_error is None
        ):
            raise RuntimeError(
                "Prediction interval model "
                "has not been fitted."
            )

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        lower = (
            predictions
            + self.lower_error
        )

        upper = (
            predictions
            + self.upper_error
        )

        lower = np.maximum(
            lower,
            0,
        )

        upper = np.maximum(
            upper,
            0,
        )

        return pd.DataFrame(
            {
                "prediction": predictions,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )