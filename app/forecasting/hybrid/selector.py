from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from app.forecasting.hybrid.schemas import (
    HybridDecision,
    ModelValidationScore,
)


class DynamicModelSelector:

    """
    Selects or weights candidate forecasting models
    using validation-time performance.

    No future/test actuals are used.
    """

    def __init__(
        self,
        minimum_weight: float = 0.05,
        error_power: float = 1.0,
        stability_power: float = 0.25,
    ) -> None:

        self.minimum_weight = (
            float(minimum_weight)
        )

        self.error_power = (
            float(error_power)
        )

        self.stability_power = (
            float(stability_power)
        )

    def select(
        self,
        scores: Iterable[
            ModelValidationScore
        ],
        horizon: int = 1,
    ) -> HybridDecision:

        scores = list(scores)

        if not scores:

            raise ValueError(
                "At least one model score is required"
            )

        for score in scores:

            if score.mae < 0:
                raise ValueError(
                    "MAE cannot be negative"
                )

            if score.rmse < 0:
                raise ValueError(
                    "RMSE cannot be negative"
                )

            if not np.isfinite(
                score.mae
            ):

                raise ValueError(
                    "MAE must be finite"
                )

            if not np.isfinite(
                score.rmse
            ):

                raise ValueError(
                    "RMSE must be finite"
                )

            if score.stability <= 0:

                raise ValueError(
                    "Stability must be positive"
                )

        # --------------------------------------------------------
        # Horizon penalty
        #
        # Longer horizons generally require greater caution.
        # We do NOT invent a fixed model preference. Instead,
        # validation error receives progressively more influence.
        # --------------------------------------------------------

        horizon_factor = max(
            1.0,
            float(horizon) ** 0.15,
        )

        raw_weights: Dict[
            str,
            float,
        ] = {}

        for score in scores:

            effective_error = (
                max(
                    score.mae,
                    1e-8,
                )
                *
                horizon_factor
            )

            stability_factor = max(
                score.stability,
                1e-6,
            )

            weight = (
                1.0
                /
                (
                    effective_error
                    **
                    self.error_power
                )
            )

            weight *= (
                stability_factor
                **
                self.stability_power
            )

            raw_weights[
                score.model_name
            ] = float(weight)

        total = sum(
            raw_weights.values()
        )

        if total <= 0:

            raise ValueError(
                "Unable to construct model weights"
            )

        weights = {
            name: value / total
            for name, value
            in raw_weights.items()
        }

        # --------------------------------------------------------
        # Remove negligible models and renormalize.
        # --------------------------------------------------------

        filtered = {
            name: weight
            for name, weight
            in weights.items()
            if weight >= self.minimum_weight
        }

        if not filtered:

            best = min(
                scores,
                key=lambda x: x.mae,
            )

            filtered = {
                best.model_name: 1.0
            }

        else:

            total_filtered = sum(
                filtered.values()
            )

            filtered = {
                name: weight / total_filtered
                for name, weight
                in filtered.items()
            }

        selected_model = max(
            filtered,
            key=filtered.get,
        )

        # --------------------------------------------------------
        # Confidence
        #
        # High concentration means the selector has stronger
        # evidence for one model.
        # --------------------------------------------------------

        concentration = max(
            filtered.values()
        )

        entropy = 0.0

        for weight in filtered.values():

            entropy -= (
                weight
                *
                np.log(
                    max(
                        weight,
                        1e-12,
                    )
                )
            )

        max_entropy = (
            np.log(
                max(
                    len(filtered),
                    1,
                )
            )
        )

        if max_entropy > 0:

            normalized_entropy = (
                entropy
                /
                max_entropy
            )

        else:

            normalized_entropy = 0.0

        confidence = (
            0.70 * concentration
            +
            0.30 * (
                1.0
                -
                normalized_entropy
            )
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        if len(filtered) == 1:

            reason = (
                "Single model dominates "
                "validation-weighted routing"
            )

        else:

            reason = (
                "Validation-weighted hybrid "
                "across multiple candidate models"
            )

        return HybridDecision(
            selected_model=selected_model,
            weights=filtered,
            confidence=confidence,
            reason=reason,
        )