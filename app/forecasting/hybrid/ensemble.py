from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


class WeightedHybridEnsemble:

    """
    Combines candidate model forecasts.

    Every candidate must predict the same horizon.
    """

    def combine(
        self,
        forecasts: Mapping[
            str,
            Sequence[float],
        ],
        weights: Mapping[
            str,
            float,
        ],
    ) -> np.ndarray:

        if not forecasts:

            raise ValueError(
                "No candidate forecasts supplied"
            )

        if not weights:

            raise ValueError(
                "No model weights supplied"
            )

        selected = []

        for model_name, weight in weights.items():

            if model_name not in forecasts:

                raise ValueError(
                    f"Missing forecast for model: "
                    f"{model_name}"
                )

            if weight < 0:

                raise ValueError(
                    "Model weights cannot be negative"
                )

            values = np.asarray(
                forecasts[model_name],
                dtype=float,
            )

            if values.ndim != 1:

                raise ValueError(
                    "Each forecast must be one-dimensional"
                )

            if not np.isfinite(
                values
            ).all():

                raise ValueError(
                    "Forecast contains non-finite values"
                )

            selected.append(
                (
                    model_name,
                    float(weight),
                    values,
                )
            )

        if not selected:

            raise ValueError(
                "No valid forecast candidates"
            )

        lengths = {
            len(values)
            for _, _, values
            in selected
        }

        if len(lengths) != 1:

            raise ValueError(
                "All forecasts must have equal horizon"
            )

        total_weight = sum(
            weight
            for _, weight, _
            in selected
        )

        if total_weight <= 0:

            raise ValueError(
                "Total weight must be positive"
            )

        horizon = len(
            selected[0][2]
        )

        result = np.zeros(
            horizon,
            dtype=float,
        )

        for _, weight, values in selected:

            result += (
                weight
                /
                total_weight
            ) * values

        return np.maximum(
            result,
            0.0,
        )