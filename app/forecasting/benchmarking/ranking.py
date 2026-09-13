from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_score(
    values: pd.Series,
    higher_is_better: bool,
) -> pd.Series:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum <= 1e-12:
        return pd.Series(
            1.0,
            index=values.index,
        )

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    if higher_is_better:
        return normalized

    return 1.0 - normalized


def rank_models(
    results: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "model_name",
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "spike_recall",
        "stability",
        "prediction_cost",
    }

    missing = (
        required
        - set(results.columns)
    )

    if missing:
        raise ValueError(
            f"benchmark results missing: "
            f"{sorted(missing)}"
        )

    ranked = results.copy()

    ranked["error_score"] = (
        0.30
        * minmax_score(
            ranked["mae"],
            higher_is_better=False,
        )
        + 0.20
        * minmax_score(
            ranked["rmse"],
            higher_is_better=False,
        )
        + 0.15
        * minmax_score(
            ranked["smape"],
            higher_is_better=False,
        )
        + 0.10
        * minmax_score(
            ranked["high_demand_mae"],
            higher_is_better=False,
        )
    )

    ranked["behavior_score"] = (
        0.10
        * minmax_score(
            ranked["spike_recall"],
            higher_is_better=True,
        )
        + 0.05
        * minmax_score(
            ranked["stability"],
            higher_is_better=True,
        )
    )

    ranked["cost_score"] = (
        0.10
        * minmax_score(
            ranked["prediction_cost"],
            higher_is_better=False,
        )
    )

    ranked["score"] = (
        ranked["error_score"]
        + ranked["behavior_score"]
        + ranked["cost_score"]
    )

    ranked["rank"] = (
        ranked["score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return ranked.sort_values(
        [
            "rank",
            "mae",
        ]
    ).reset_index(
        drop=True
    )