from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from app.forecasting.hybrid.ensemble import (
    WeightedHybridEnsemble,
)

from app.forecasting.hybrid.schemas import (
    HybridForecast,
    HybridDecision,
    ModelValidationScore,
)

from app.forecasting.hybrid.selector import (
    DynamicModelSelector,
)


class HybridForecastService:

    """
    Unified Hybrid ML service.

    Flow:

        candidate forecasts
              +
        validation scores
              ↓
        model selector
              ↓
        weighted ensemble
              ↓
        final forecast
    """

    def __init__(
        self,
        minimum_weight: float = 0.05,
        error_power: float = 1.0,
        stability_power: float = 0.25,
    ) -> None:

        self.selector = (
            DynamicModelSelector(
                minimum_weight=minimum_weight,
                error_power=error_power,
                stability_power=stability_power,
            )
        )

        self.ensemble = (
            WeightedHybridEnsemble()
        )

    def forecast(
        self,
        candidate_forecasts: Mapping[
            str,
            Sequence[float],
        ],
        validation_scores: Sequence[
            ModelValidationScore
        ],
        horizon: int,
    ) -> HybridForecast:

        if horizon <= 0:

            raise ValueError(
                "Horizon must be positive"
            )

        decision = (
            self.selector.select(
                validation_scores,
                horizon=horizon,
            )
        )

        predictions = (
            self.ensemble.combine(
                candidate_forecasts,
                decision.weights,
            )
        )

        if len(predictions) != horizon:

            raise ValueError(
                "Hybrid forecast horizon mismatch"
            )

        return HybridForecast(
            predictions=predictions.tolist(),
            selected_model=(
                decision.selected_model
            ),
            weights=decision.weights,
            confidence=decision.confidence,
        )

    def decision(
        self,
        validation_scores: Sequence[
            ModelValidationScore
        ],
        horizon: int,
    ) -> HybridDecision:

        return self.selector.select(
            validation_scores,
            horizon=horizon,
        )

    def validate(
        self,
        result: HybridForecast,
        horizon: int,
    ) -> dict:

        errors = []

        if horizon <= 0:

            errors.append(
                "Horizon must be positive"
            )

        if len(
            result.predictions
        ) != horizon:

            errors.append(
                "Prediction length does not match horizon"
            )

        if not np.isfinite(
            np.asarray(
                result.predictions,
                dtype=float,
            )
        ).all():

            errors.append(
                "Predictions contain non-finite values"
            )

        if not (
            0.0
            <= result.confidence
            <= 1.0
        ):

            errors.append(
                "Confidence must be in [0, 1]"
            )

        weights = result.weights

        if not weights:

            errors.append(
                "Weights cannot be empty"
            )

        else:

            if any(
                weight < 0
                for weight
                in weights.values()
            ):

                errors.append(
                    "Weights cannot be negative"
                )

            total = sum(
                weights.values()
            )

            if not np.isclose(
                total,
                1.0,
                atol=1e-6,
            ):

                errors.append(
                    "Weights must sum to 1"
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }