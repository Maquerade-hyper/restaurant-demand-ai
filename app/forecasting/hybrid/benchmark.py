from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from app.forecasting.hybrid.schemas import ModelValidationScore
from app.forecasting.hybrid.selector import DynamicModelSelector
from app.forecasting.hybrid.ensemble import WeightedHybridEnsemble


class GenuineHybridBenchmark:
    """
    Prediction-level hybrid benchmark.

    IMPORTANT:
    Validation predictions are used only to derive model weights.
    Test predictions are never used by the selector.
    """

    REQUIRED_MODELS = ("naive", "xgboost", "transformer")

    def __init__(self):
        self.selector = DynamicModelSelector()
        self.ensemble = WeightedHybridEnsemble()

    @staticmethod
    def _validate_vector(name: str, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=float)

        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

        if len(arr) == 0:
            raise ValueError(f"{name} cannot be empty")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

        return arr

    @staticmethod
    def _mae(actual, predicted):
        return float(np.mean(np.abs(actual - predicted)))

    @staticmethod
    def _rmse(actual, predicted):
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    @staticmethod
    def _smape(actual, predicted):
        denominator = np.abs(actual) + np.abs(predicted)
        values = np.where(
            denominator == 0,
            0.0,
            2.0 * np.abs(actual - predicted) / denominator,
        )
        return float(np.mean(values) * 100.0)

    @staticmethod
    def _bias(actual, predicted):
        return float(np.mean(predicted - actual))

    @staticmethod
    def _high_demand_mae(actual, predicted):
        threshold = float(np.quantile(actual, 0.75))
        mask = actual >= threshold

        if not np.any(mask):
            return 0.0

        return float(np.mean(np.abs(actual[mask] - predicted[mask])))

    @staticmethod
    def _spike_recall(actual, predicted):
        """
        Conservative spike recall.

        A spike is an actual value >= 90th percentile.
        A prediction counts as detecting it when it is >=
        the 75th percentile of the actual test distribution.
        """
        threshold_actual = float(np.quantile(actual, 0.90))
        threshold_pred = float(np.quantile(actual, 0.75))

        actual_spikes = actual >= threshold_actual

        if not np.any(actual_spikes):
            return 0.0

        detected = predicted[actual_spikes] >= threshold_pred
        return float(np.mean(detected))

    def _metrics(self, actual, predicted, model_name):
        return {
            "model_name": model_name,
            "mae": self._mae(actual, predicted),
            "rmse": self._rmse(actual, predicted),
            "smape": self._smape(actual, predicted),
            "bias": self._bias(actual, predicted),
            "high_demand_mae": self._high_demand_mae(
                actual,
                predicted,
            ),
            "spike_recall": self._spike_recall(
                actual,
                predicted,
            ),
        }

    def derive_validation_scores(
        self,
        actual_validation,
        validation_predictions: Dict[str, Iterable[float]],
    ):
        actual = self._validate_vector(
            "actual_validation",
            actual_validation,
        )

        scores = {}

        for model_name in self.REQUIRED_MODELS:
            if model_name not in validation_predictions:
                raise ValueError(
                    f"Missing validation predictions for {model_name}"
                )

            prediction = self._validate_vector(
                f"{model_name}_validation",
                validation_predictions[model_name],
            )

            if len(prediction) != len(actual):
                raise ValueError(
                    f"Validation length mismatch for {model_name}"
                )

            scores[model_name] = ModelValidationScore(
                model_name=model_name,
                mae=self._mae(actual, prediction),
                rmse=self._rmse(actual, prediction),
                bias=self._bias(actual, prediction),
                stability=1.0,
            )

        return scores

    def evaluate(
        self,
        actual_validation,
        validation_predictions,
        actual_test,
        test_predictions,
        horizon: int,
    ):
        """
        Complete genuine hybrid evaluation.

        Validation:
            used to calculate weights.

        Test:
            used only after weights are frozen.
        """
        actual_val = self._validate_vector(
            "actual_validation",
            actual_validation,
        )

        actual_test = self._validate_vector(
            "actual_test",
            actual_test,
        )

        validation_scores = self.derive_validation_scores(
            actual_val,
            validation_predictions,
        )

        score_list = list(validation_scores.values())

        decision = self.selector.select(
            score_list,
            horizon=horizon,
        )

        clean_test_predictions = {}

        for model_name in self.REQUIRED_MODELS:
            if model_name not in test_predictions:
                raise ValueError(
                    f"Missing test predictions for {model_name}"
                )

            prediction = self._validate_vector(
                f"{model_name}_test",
                test_predictions[model_name],
            )

            if len(prediction) != len(actual_test):
                raise ValueError(
                    f"Test length mismatch for {model_name}"
                )

            clean_test_predictions[model_name] = prediction

        hybrid_prediction = self.ensemble.combine(
            clean_test_predictions,
            decision.weights,
        )

        rows = []

        for model_name in self.REQUIRED_MODELS:
            rows.append(
                self._metrics(
                    actual_test,
                    clean_test_predictions[model_name],
                    model_name,
                )
            )

        rows.append(
            self._metrics(
                actual_test,
                hybrid_prediction,
                "hybrid",
            )
        )

        result = pd.DataFrame(rows)

        best_individual = float(
            result.loc[
                result["model_name"].isin(self.REQUIRED_MODELS),
                "mae",
            ].min()
        )

        hybrid_mae = float(
            result.loc[
                result["model_name"] == "hybrid",
                "mae",
            ].iloc[0]
        )

        result["improvement_vs_best_individual_pct"] = (
            (best_individual - result["mae"])
            / max(best_individual, 1e-12)
            * 100.0
        )

        result["hybrid_beats_all"] = (
            hybrid_mae < best_individual
        )

        return {
            "metrics": result,
            "validation_scores": validation_scores,
            "decision": decision,
            "hybrid_predictions": hybrid_prediction,
            "best_individual_mae": best_individual,
            "hybrid_mae": hybrid_mae,
            "hybrid_beats_all": hybrid_mae < best_individual,
        }

    @staticmethod
    def validate(result):
        errors = []

        if not isinstance(result, dict):
            return {
                "passed": False,
                "errors": ["result must be a dictionary"],
            }

        required = {
            "metrics",
            "validation_scores",
            "decision",
            "hybrid_predictions",
        }

        missing = required - set(result.keys())

        if missing:
            errors.append(
                f"missing result fields: {sorted(missing)}"
            )

        if "metrics" in result:
            metrics = result["metrics"]

            required_columns = {
                "model_name",
                "mae",
                "rmse",
                "smape",
                "bias",
                "high_demand_mae",
                "spike_recall",
            }

            missing_columns = (
                required_columns - set(metrics.columns)
            )

            if missing_columns:
                errors.append(
                    "missing metric columns: "
                    f"{sorted(missing_columns)}"
                )

            models = set(metrics["model_name"])

            if not set(
                GenuineHybridBenchmark.REQUIRED_MODELS
            ).issubset(models):
                errors.append(
                    "individual model results missing"
                )

            if "hybrid" not in models:
                errors.append(
                    "hybrid result missing"
                )

            if (metrics["mae"] < 0).any():
                errors.append(
                    "negative MAE detected"
                )

            if (metrics["rmse"] < 0).any():
                errors.append(
                    "negative RMSE detected"
                )

        prediction = result.get("hybrid_predictions")

        if prediction is not None:
            arr = np.asarray(prediction, dtype=float)

            if arr.ndim != 1:
                errors.append(
                    "hybrid_predictions must be 1D"
                )

            elif not np.all(np.isfinite(arr)):
                errors.append(
                    "hybrid_predictions contain non-finite values"
                )

            elif np.any(arr < 0):
                errors.append(
                    "hybrid_predictions contain negative values"
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }