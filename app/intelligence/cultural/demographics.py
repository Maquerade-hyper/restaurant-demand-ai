from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
}


def prepare_demographics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"demographics missing columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "demographics is empty"
        )

    result = df.copy()

    # ---------------------------------------------------------
    # Demographic signals are normalized indicators.
    #
    # We deliberately do NOT convert demographic identity
    # directly into a demand assumption.
    #
    # Example:
    # religious_population_share = measured local population
    # characteristic.
    #
    # The demand engine learns whether that signal is actually
    # associated with observed demand.
    # ---------------------------------------------------------

    numeric_candidates = [
        "population_density",
        "tourism_index",
        "student_index",
        "business_index",
        "residential_index",
        "religious_population_share",
        "young_population_share",
        "working_population_share",
    ]

    for column in numeric_candidates:

        if column not in result.columns:
            result[column] = 0.0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

        result[column] = (
            result[column]
            .clip(lower=0.0)
        )

    # ---------------------------------------------------------
    # Optional categorical descriptors.
    # ---------------------------------------------------------

    for column in [
        "dominant_religious_context",
        "demographic_context",
        "city_class",
    ]:

        if column not in result.columns:
            result[column] = "unknown"

        result[column] = (
            result[column]
            .fillna("unknown")
            .astype(str)
        )

    return result