from __future__ import annotations

import numpy as np
import pandas as pd


def build_cultural_effects(
    demand: pd.DataFrame,
    calendar: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:

    required_demand = {
        "outlet_id",
        "product_id",
        "date",
        "demand",
    }

    missing = (
        required_demand
        - set(demand.columns)
    )

    if missing:
        raise ValueError(
            f"demand missing columns: {sorted(missing)}"
        )

    result = demand.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    # ---------------------------------------------------------
    # Calendar can be outlet-specific or global.
    # ---------------------------------------------------------

    calendar_keys = [
        column
        for column in ["outlet_id", "date"]
        if column in calendar.columns
    ]

    if calendar_keys == ["date"]:

        result = result.merge(
            calendar,
            on="date",
            how="left",
        )

    else:

        result = result.merge(
            calendar,
            on=["outlet_id", "date"],
            how="left",
        )

    # ---------------------------------------------------------
    # Demographics are normally outlet-level.
    # ---------------------------------------------------------

    demographic_columns = [
        "outlet_id",
        "population_density",
        "tourism_index",
        "student_index",
        "business_index",
        "residential_index",
        "religious_population_share",
        "young_population_share",
        "working_population_share",
        "dominant_religious_context",
        "demographic_context",
        "city_class",
    ]

    available = [
        column
        for column in demographic_columns
        if column in demographics.columns
    ]

    result = result.merge(
        demographics[available].drop_duplicates(
            subset=["outlet_id"]
        ),
        on="outlet_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Fill missing context.
    # ---------------------------------------------------------

    numeric_columns = [
        "holiday_importance",
        "religious_importance",
        "event_importance",
        "population_density",
        "tourism_index",
        "student_index",
        "business_index",
        "residential_index",
        "religious_population_share",
        "young_population_share",
        "working_population_share",
    ]

    for column in numeric_columns:

        if column not in result.columns:
            result[column] = 0.0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    # ---------------------------------------------------------
    # Context interactions.
    #
    # These are signals, not predetermined demand multipliers.
    # ---------------------------------------------------------

    result["religious_context_exposure"] = (
        result["religious_importance"]
        * result["religious_population_share"]
    )

    result["tourism_context_exposure"] = (
        result["tourism_index"]
        * result["holiday_importance"]
    )

    result["student_context_exposure"] = (
        result["student_index"]
        * result["holiday_importance"]
    )

    result["business_context_exposure"] = (
        result["business_index"]
        * result["holiday_importance"]
    )

    result["working_population_exposure"] = (
        result["working_population_share"]
        * result["business_index"]
    )

    # ---------------------------------------------------------
    # Composite cultural pressure.
    #
    # This is intentionally bounded.
    # ---------------------------------------------------------

    result["cultural_context_score"] = (
        0.40 * result["religious_context_exposure"]
        + 0.25 * result["holiday_importance"]
        + 0.20 * result["event_importance"]
        + 0.15 * result["tourism_context_exposure"]
    ).clip(0.0, 1.0)

    # ---------------------------------------------------------
    # Context-adjusted demand reference.
    #
    # This is NOT a causal forecast.
    #
    # It is a diagnostic/context feature derived from the
    # historical demand and context signals.
    # ---------------------------------------------------------

    result = result.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    )

    result["historical_demand_reference"] = (
        result
        .groupby(
            ["outlet_id", "product_id"],
            sort=False,
        )["demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    28,
                    min_periods=7,
                )
                .mean()
            )
        )
    )

    result["historical_demand_reference"] = (
        result["historical_demand_reference"]
        .fillna(
            result["demand"]
            .shift(1)
        )
        .fillna(
            result["demand"]
        )
        .clip(lower=0.0)
    )

    # ---------------------------------------------------------
    # Context uplift signal.
    #
    # This does NOT claim that cultural context causes a fixed
    # percentage increase.
    #
    # It simply expresses the context intensity relative to the
    # historical demand reference.
    # ---------------------------------------------------------

    result["cultural_demand_pressure"] = (
        result["historical_demand_reference"]
        * result["cultural_context_score"]
    )

    result["cultural_demand_pressure"] = (
        result["cultural_demand_pressure"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return result