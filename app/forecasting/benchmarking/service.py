from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from app.forecasting.baseline import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
)


class ModelBenchmarkingService:
    """
    Part 21 - Model Benchmarking

    Provides a common evaluation contract for forecasting models.

    Built-in models:
        - naive
        - seasonal_naive
        - moving_average

    Optional:
        - xgboost predictions supplied externally

    Output:
        pandas.DataFrame

    This keeps Part 21 independent from model training.
    """

    FORBIDDEN_COLUMNS = {
        "true_demand",
        "lost_demand_truth",
        "demand_truth",
        "future_demand",
    }

    # ------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------

    @staticmethod
    def _to_series(values) -> pd.Series:
        """
        Part 8 baseline functions require pandas.Series.
        """

        if isinstance(values, pd.Series):
            return values.reset_index(drop=True).astype(float)

        if isinstance(values, pd.DataFrame):

            if values.shape[1] != 1:
                raise ValueError(
                    "DataFrame input must contain exactly one column"
                )

            return pd.Series(
                values.iloc[:, 0].to_numpy(dtype=float)
            ).reset_index(drop=True)

        return pd.Series(
            np.asarray(values, dtype=float)
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # BASELINE PREDICTIONS
    # ------------------------------------------------------------

    def _run_naive(
        self,
        history: pd.Series,
        horizon: int,
    ) -> tuple[np.ndarray, float]:

        start = time.perf_counter()

        prediction = naive_forecast(
            history=history,
            horizon=horizon,
        )

        elapsed = time.perf_counter() - start

        return (
            np.asarray(prediction, dtype=float),
            elapsed,
        )

    def _run_seasonal_naive(
        self,
        history: pd.Series,
        horizon: int,
    ) -> tuple[np.ndarray, float]:

        start = time.perf_counter()

        prediction = seasonal_naive_forecast(
            history=history,
            horizon=horizon,
        )

        elapsed = time.perf_counter() - start

        return (
            np.asarray(prediction, dtype=float),
            elapsed,
        )

    def _run_moving_average(
        self,
        history: pd.Series,
        horizon: int,
    ) -> tuple[np.ndarray, float]:

        start = time.perf_counter()

        prediction = moving_average_forecast(
            history=history,
            horizon=horizon,
        )

        elapsed = time.perf_counter() - start

        return (
            np.asarray(prediction, dtype=float),
            elapsed,
        )

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    @staticmethod
    def _mae(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        return float(
            np.mean(
                np.abs(actual - prediction)
            )
        )

    @staticmethod
    def _rmse(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        actual - prediction
                    )
                )
            )
        )

    @staticmethod
    def _smape(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        denominator = (
            np.abs(actual) +
            np.abs(prediction)
        )

        mask = denominator > 0

        if not np.any(mask):
            return 0.0

        return float(
            np.mean(
                2.0
                * np.abs(
                    actual[mask] -
                    prediction[mask]
                )
                / denominator[mask]
            )
            * 100.0
        )

    @staticmethod
    def _bias(
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        return float(
            np.mean(
                prediction - actual
            )
        )

    @staticmethod
    def _high_demand_mask(
        actual: np.ndarray,
        quantile: float = 0.80,
    ) -> np.ndarray:

        threshold = float(
            np.quantile(
                actual,
                quantile,
            )
        )

        return actual >= threshold

    @classmethod
    def _high_demand_mae(
        cls,
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        mask = cls._high_demand_mask(actual)

        if not np.any(mask):
            return 0.0

        return float(
            np.mean(
                np.abs(
                    actual[mask] -
                    prediction[mask]
                )
            )
        )

    @classmethod
    def _high_demand_bias(
        cls,
        actual: np.ndarray,
        prediction: np.ndarray,
    ) -> float:

        mask = cls._high_demand_mask(actual)

        if not np.any(mask):
            return 0.0

        return float(
            np.mean(
                prediction[mask] -
                actual[mask]
            )
        )

    @staticmethod
    def _spike_recall(
        actual: np.ndarray,
        prediction: np.ndarray,
        quantile: float = 0.90,
    ) -> float:

        actual_threshold = float(
            np.quantile(
                actual,
                quantile,
            )
        )

        prediction_threshold = float(
            np.quantile(
                prediction,
                quantile,
            )
        )

        actual_spikes = (
            actual >= actual_threshold
        )

        predicted_spikes = (
            prediction >= prediction_threshold
        )

        count = int(
            np.sum(actual_spikes)
        )

        if count == 0:
            return 0.0

        return float(
            np.sum(
                actual_spikes &
                predicted_spikes
            )
            / count
        )

    @staticmethod
    def _stability(
        prediction: np.ndarray,
    ) -> float:

        if len(prediction) <= 1:
            return 1.0

        differences = np.diff(
            prediction
        )

        scale = float(
            np.mean(
                np.abs(prediction)
            )
        )

        if scale <= 1e-12:
            scale = 1.0

        volatility = float(
            np.std(differences) /
            scale
        )

        return float(
            1.0 /
            (1.0 + volatility)
        )

    # ------------------------------------------------------------
    # SINGLE MODEL EVALUATION
    # ------------------------------------------------------------

    def _evaluate_model(
        self,
        model_name: str,
        actual,
        prediction,
        elapsed_seconds: float,
    ) -> dict:

        actual = self._to_series(
            actual
        ).to_numpy()

        prediction = self._to_series(
            prediction
        ).to_numpy()

        if len(actual) != len(prediction):
            raise ValueError(
                f"{model_name}: prediction length "
                f"{len(prediction)} != actual length "
                f"{len(actual)}"
            )

        if not np.all(
            np.isfinite(prediction)
        ):
            raise ValueError(
                f"{model_name}: non-finite prediction"
            )

        mae = self._mae(
            actual,
            prediction,
        )

        rmse = self._rmse(
            actual,
            prediction,
        )

        smape = self._smape(
            actual,
            prediction,
        )

        bias = self._bias(
            actual,
            prediction,
        )

        high_mae = self._high_demand_mae(
            actual,
            prediction,
        )

        high_bias = self._high_demand_bias(
            actual,
            prediction,
        )

        spike_recall = self._spike_recall(
            actual,
            prediction,
        )

        stability = self._stability(
            prediction
        )

        prediction_cost = float(
            elapsed_seconds /
            max(len(prediction), 1)
        )

        return {
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "bias": bias,
            "high_demand_mae": high_mae,
            "high_demand_bias": high_bias,
            "spike_recall": spike_recall,
            "stability": stability,
            "prediction_cost": prediction_cost,
        }

    # ------------------------------------------------------------
    # NORMALIZATION FOR RANKING
    # ------------------------------------------------------------

    @staticmethod
    def _score_lower_is_better(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=float,
        )

        low = float(
            np.min(values)
        )

        high = float(
            np.max(values)
        )

        if abs(high - low) < 1e-12:
            return np.ones(
                len(values),
                dtype=float,
            )

        return (
            high - values
        ) / (
            high - low
        )

    @staticmethod
    def _score_higher_is_better(
        values: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            values,
            dtype=float,
        )

        low = float(
            np.min(values)
        )

        high = float(
            np.max(values)
        )

        if abs(high - low) < 1e-12:
            return np.ones(
                len(values),
                dtype=float,
            )

        return (
            values - low
        ) / (
            high - low
        )

    # ------------------------------------------------------------
    # RANK MODELS
    # ------------------------------------------------------------

    def _rank(
        self,
        results: list[dict],
    ) -> pd.DataFrame:

        frame = pd.DataFrame(
            results
        )

        if frame.empty:
            return frame

        error_score = (
            0.30
            * self._score_lower_is_better(
                frame["mae"].to_numpy()
            )
            +
            0.20
            * self._score_lower_is_better(
                frame["rmse"].to_numpy()
            )
            +
            0.15
            * self._score_lower_is_better(
                frame["smape"].to_numpy()
            )
            +
            0.10
            * self._score_lower_is_better(
                frame[
                    "high_demand_mae"
                ].to_numpy()
            )
        )

        behavior_score = (
            0.10
            * self._score_higher_is_better(
                frame[
                    "spike_recall"
                ].to_numpy()
            )
            +
            0.05
            * self._score_higher_is_better(
                frame[
                    "stability"
                ].to_numpy()
            )
        )

        cost_score = (
            0.10
            * self._score_lower_is_better(
                frame[
                    "prediction_cost"
                ].to_numpy()
            )
        )

        frame["error_score"] = (
            error_score
        )

        frame["behavior_score"] = (
            behavior_score
        )

        frame["cost_score"] = (
            cost_score
        )

        frame["score"] = (
            error_score
            +
            behavior_score
            +
            cost_score
        )

        # Backward-compatible internal name
        frame["benchmark_score"] = frame["score"]

        frame = frame.sort_values(
            by=[
                "score",
                "mae",
                "rmse",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).reset_index(
            drop=True
        )

        frame["rank"] = (
            np.arange(
                1,
                len(frame) + 1,
            )
        )

        return frame

    # ------------------------------------------------------------
    # MAIN BENCHMARK
    # ------------------------------------------------------------

    def analyze(
        self,
        history,
        test,
        xgb_actual=None,
        xgb_prediction=None,
        xgb_prediction_cost: float = 0.0,
    ) -> pd.DataFrame:

        history = self._to_series(
            history
        )

        test = self._to_series(
            test
        )

        if history.empty:
            raise ValueError(
                "History cannot be empty"
            )

        if test.empty:
            raise ValueError(
                "Test cannot be empty"
            )

        horizon = len(test)

        results = []

        # --------------------------------------------------------
        # NAIVE
        # --------------------------------------------------------

        prediction, elapsed = (
            self._run_naive(
                history,
                horizon,
            )
        )

        results.append(
            self._evaluate_model(
                model_name="naive",
                actual=test,
                prediction=prediction,
                elapsed_seconds=elapsed,
            )
        )

        # --------------------------------------------------------
        # SEASONAL NAIVE
        # --------------------------------------------------------

        prediction, elapsed = (
            self._run_seasonal_naive(
                history,
                horizon,
            )
        )

        results.append(
            self._evaluate_model(
                model_name="seasonal_naive",
                actual=test,
                prediction=prediction,
                elapsed_seconds=elapsed,
            )
        )

        # --------------------------------------------------------
        # MOVING AVERAGE
        # --------------------------------------------------------

        prediction, elapsed = (
            self._run_moving_average(
                history,
                horizon,
            )
        )

        results.append(
            self._evaluate_model(
                model_name="moving_average",
                actual=test,
                prediction=prediction,
                elapsed_seconds=elapsed,
            )
        )

        # --------------------------------------------------------
        # OPTIONAL XGBOOST
        # --------------------------------------------------------

        if xgb_prediction is not None:

            xgb_prediction = (
                self._to_series(
                    xgb_prediction
                )
            )

            if len(xgb_prediction) != horizon:
                raise ValueError(
                    "xgb_prediction length must "
                    "equal test horizon"
                )

            if xgb_actual is None:
                xgb_actual = test
            else:
                xgb_actual = (
                    self._to_series(
                        xgb_actual
                    )
                )

            if len(xgb_actual) != horizon:
                raise ValueError(
                    "xgb_actual length must "
                    "equal test horizon"
                )

            results.append(
                self._evaluate_model(
                    model_name="xgboost",
                    actual=xgb_actual,
                    prediction=xgb_prediction,
                    elapsed_seconds=float(
                        xgb_prediction_cost
                    ),
                )
            )

        return self._rank(
            results
        )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    def validate(
        self,
        result: pd.DataFrame,
    ) -> dict:

        errors = []

        # Must be DataFrame
        if not isinstance(
            result,
            pd.DataFrame,
        ):
            errors.append(
                "Result must be pandas.DataFrame"
            )
            return {
                "passed": False,
                "errors": errors,
            }

        # Must not be empty
        if result.empty:
            errors.append(
                "Benchmark result is empty"
            )

        required_columns = {
            "model_name",
            "mae",
            "rmse",
            "smape",
            "bias",
            "high_demand_mae",
            "high_demand_bias",
            "spike_recall",
            "stability",
            "prediction_cost",
            "error_score",
            "behavior_score",
            "cost_score",
            "score",
            "benchmark_score",
            "rank",
        }

        missing = (
            required_columns -
            set(result.columns)
        )

        if missing:
            errors.append(
                "Missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        # Truth/leakage protection
        forbidden_present = (
            self.FORBIDDEN_COLUMNS &
            set(result.columns)
        )

        if forbidden_present:
            errors.append(
                "Forbidden truth columns present: "
                + ", ".join(
                    sorted(forbidden_present)
                )
            )

        # Numeric columns
        numeric_columns = (
            required_columns -
            {"model_name"}
        )

        for column in numeric_columns:

            if column not in result.columns:
                continue

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if values.isna().any():
                errors.append(
                    f"{column} contains non-numeric values"
                )

            if not np.all(
                np.isfinite(
                    values.to_numpy()
                )
            ):
                errors.append(
                    f"{column} contains non-finite values"
                )

        # Metric bounds
        bounded_columns = {
            "spike_recall",
            "stability",
        }

        for column in bounded_columns:

            if column not in result.columns:
                continue

            values = result[column]

            if (
                (values < 0).any()
                or
                (values > 1).any()
            ):
                errors.append(
                    f"{column} outside [0, 1]"
                )

        # Error metrics cannot be negative
        non_negative_columns = {
            "mae",
            "rmse",
            "smape",
            "high_demand_mae",
            "prediction_cost",
        }

        for column in non_negative_columns:

            if column not in result.columns:
                continue

            if (
                result[column] < 0
            ).any():

                errors.append(
                    f"{column} contains negative values"
                )

        # Rank integrity
        if "rank" in result.columns:

            ranks = result[
                "rank"
            ].tolist()

            expected = list(
                range(
                    1,
                    len(result) + 1,
                )
            )

            if sorted(ranks) != expected:
                errors.append(
                    "Ranks are not unique and consecutive"
                )

        # Model names
        if "model_name" in result.columns:

            if (
                result["model_name"]
                .duplicated()
                .any()
            ):
                errors.append(
                    "Duplicate model names found"
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rows": int(len(result)),
            "models": (
                result["model_name"]
                .tolist()
                if "model_name" in result.columns
                else []
            ),
        }