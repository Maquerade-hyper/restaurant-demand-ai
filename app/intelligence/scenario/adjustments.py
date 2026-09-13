from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIO_COLUMNS = [
    "holiday_multiplier",
    "promotion_multiplier",
    "event_multiplier",
    "weather_multiplier",
    "tourism_multiplier",
    "demographic_multiplier",
    "cultural_multiplier",
]


def validate_multiplier(
    value: float,
    minimum: float = 0.50,
    maximum: float = 2.00,
) -> float:

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            "scenario multiplier must be finite"
        )

    if value < minimum or value > maximum:
        raise ValueError(
            f"scenario multiplier must be between "
            f"{minimum} and {maximum}"
        )

    return value


def combined_multiplier(
    multipliers: dict[str, float],
) -> float:

    result = 1.0

    for value in multipliers.values():

        result *= validate_multiplier(
            value
        )

    return float(result)


def apply_scenario(
    df: pd.DataFrame,
    baseline_column: str = "baseline_demand",
) -> pd.DataFrame:

    if baseline_column not in df.columns:
        raise ValueError(
            f"missing baseline column: "
            f"{baseline_column}"
        )

    result = df.copy()

    available = []

    for column in SCENARIO_COLUMNS:

        if column not in result.columns:
            result[column] = 1.0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if result[column].isna().any():
            raise ValueError(
                f"{column} contains invalid values"
            )

        result[column] = result[column].clip(
            lower=0.50,
            upper=2.00,
        )

        available.append(column)

    result["scenario_multiplier"] = (
        result[available]
        .prod(axis=1)
    )

    result["scenario_demand"] = (
        pd.to_numeric(
            result[baseline_column],
            errors="coerce",
        )
        * result["scenario_multiplier"]
    )

    result["scenario_demand"] = (
        result["scenario_demand"]
        .clip(lower=0.0)
    )

    result["baseline_demand"] = (
        pd.to_numeric(
            result["baseline_demand"],
            errors="coerce",
        )
        .clip(lower=0.0)
    )

    result["absolute_change"] = (
        result["scenario_demand"]
        - result["baseline_demand"]
    )

    denominator = result["baseline_demand"].replace(
        0,
        np.nan,
    )

    result["percentage_change"] = (
        result["absolute_change"]
        / denominator
        * 100.0
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    return result