from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.probabilistic.quantiles import (
    build_quantile_forecast,
)

from app.forecasting.probabilistic.conformal import (
    conformal_interval,
    asymmetric_conformal_interval,
)

from app.forecasting.probabilistic.distribution import (
    build_predictive_distribution,
)

from app.forecasting.probabilistic.calibration import (
    evaluate_calibration,
)


class ProbabilisticForecastingService:
    """
    Part 24 - Unified Probabilistic Forecasting Service.

    The service separates:

        training/calibration data
                ↓
        predictive distribution
                ↓
        forecast quantiles
                ↓
        conformal intervals
                ↓
        reliability evaluation
    """

    def __init__(
        self,
        coverage: float = 0.90,
    ) -> None:

        if not (
            0.0
            <
            coverage
            <
            1.0
        ):
            raise ValueError(
                "coverage must be between 0 and 1"
            )

        self.coverage = coverage

    # ============================================================
    # QUANTILES
    # ============================================================

    def quantiles(
        self,
        point_forecast,
        calibration_actual,
        calibration_prediction,
    ) -> dict[str, np.ndarray]:

        residuals = (
            np.asarray(
                calibration_actual,
                dtype=float,
            )
            -
            np.asarray(
                calibration_prediction,
                dtype=float,
            )
        )

        return build_quantile_forecast(
            point_forecast,
            residuals,
        )

    # ============================================================
    # CONFORMAL
    # ============================================================

    def interval(
        self,
        point_forecast,
        calibration_actual,
        calibration_prediction,
    ) -> dict:

        lower, upper, radius = (
            conformal_interval(
                point_forecast,
                calibration_actual,
                calibration_prediction,
                coverage=self.coverage,
            )
        )

        return {
            "lower": lower,
            "upper": upper,
            "radius": radius,
            "coverage": self.coverage,
        }

    # ============================================================
    # ASYMMETRIC INTERVAL
    # ============================================================

    def asymmetric_interval(
        self,
        point_forecast,
        calibration_actual,
        calibration_prediction,
        lower_coverage: float = 0.10,
        upper_coverage: float = 0.90,
    ) -> dict:

        lower, upper = (
            asymmetric_conformal_interval(
                point_forecast,
                calibration_actual,
                calibration_prediction,
                lower_coverage=lower_coverage,
                upper_coverage=upper_coverage,
            )
        )

        return {
            "lower": lower,
            "upper": upper,
            "lower_coverage": lower_coverage,
            "upper_coverage": upper_coverage,
        }

    # ============================================================
    # DISTRIBUTION
    # ============================================================

    def distribution(
        self,
        point_forecast,
        calibration_actual,
        calibration_prediction,
    ) -> pd.DataFrame:

        residuals = (
            np.asarray(
                calibration_actual,
                dtype=float,
            )
            -
            np.asarray(
                calibration_prediction,
                dtype=float,
            )
        )

        return build_predictive_distribution(
            point_forecast,
            residuals,
        )

    # ============================================================
    # COMPLETE FORECAST
    # ============================================================

    def forecast(
        self,
        point_forecast,
        calibration_actual,
        calibration_prediction,
    ) -> dict:

        quantile_forecasts = self.quantiles(
            point_forecast,
            calibration_actual,
            calibration_prediction,
        )

        interval = self.interval(
            point_forecast,
            calibration_actual,
            calibration_prediction,
        )

        asymmetric = self.asymmetric_interval(
            point_forecast,
            calibration_actual,
            calibration_prediction,
        )

        distribution = self.distribution(
            point_forecast,
            calibration_actual,
            calibration_prediction,
        )

        return {
            "quantiles": quantile_forecasts,
            "interval": interval,
            "asymmetric_interval": asymmetric,
            "distribution": distribution,
        }

    # ============================================================
    # CALIBRATION EVALUATION
    # ============================================================

    def evaluate(
        self,
        actual,
        point_forecast,
        calibration_actual,
        calibration_prediction,
    ) -> dict:

        forecast = self.forecast(
            point_forecast,
            calibration_actual,
            calibration_prediction,
        )

        predictions = (
            forecast["quantiles"]
        )

        intervals = {
            "conformal_90": (
                forecast["interval"]["lower"],
                forecast["interval"]["upper"],
            ),
            "asymmetric_80": (
                forecast["asymmetric_interval"]["lower"],
                forecast["asymmetric_interval"]["upper"],
            ),
        }

        return evaluate_calibration(
            actual,
            predictions,
            intervals,
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_forecast(
        self,
        forecast: dict,
    ) -> dict:

        errors: list[str] = []

        if "quantiles" not in forecast:
            errors.append(
                "Missing quantile forecasts"
            )

        if "interval" not in forecast:
            errors.append(
                "Missing conformal interval"
            )

        if "distribution" not in forecast:
            errors.append(
                "Missing predictive distribution"
            )

        if errors:
            return {
                "passed": False,
                "errors": errors,
            }

        quantiles = forecast[
            "quantiles"
        ]

        expected_quantiles = {
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
        }

        missing = (
            expected_quantiles
            -
            set(quantiles.keys())
        )

        if missing:
            errors.append(
                "Missing quantiles: "
                +
                ", ".join(
                    sorted(missing)
                )
            )

        # --------------------------------------------------------
        # Quantile monotonicity
        # --------------------------------------------------------

        ordered = [
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
        ]

        if not missing:

            for left, right in zip(
                ordered[:-1],
                ordered[1:],
            ):

                a = np.asarray(
                    quantiles[left],
                    dtype=float,
                )

                b = np.asarray(
                    quantiles[right],
                    dtype=float,
                )

                if np.any(
                    a > b + 1e-9
                ):

                    errors.append(
                        f"Quantile crossing: "
                        f"{left} > {right}"
                    )

        # --------------------------------------------------------
        # Conformal interval
        # --------------------------------------------------------

        interval = forecast[
            "interval"
        ]

        lower = np.asarray(
            interval["lower"],
            dtype=float,
        )

        upper = np.asarray(
            interval["upper"],
            dtype=float,
        )

        if np.any(
            lower > upper + 1e-9
        ):
            errors.append(
                "Conformal lower bound exceeds upper bound"
            )

        # --------------------------------------------------------
        # Finite values
        # --------------------------------------------------------

        for name, values in quantiles.items():

            values = np.asarray(
                values,
                dtype=float,
            )

            if not np.isfinite(
                values
            ).all():

                errors.append(
                    f"Non-finite values in {name}"
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }