from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "product_id",
    "date",
    "quantity_sold",
    "deconstrained_demand",
    "estimated_lost_demand",
    "stockout",
}


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def add_lost_demand_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert deconstrained demand estimates into daily
    lost-demand intelligence.

    No ground-truth demand is required.
    """

    validate_input(df)

    result = df.copy()

    result["quantity_sold"] = (
        pd.to_numeric(
            result["quantity_sold"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result["deconstrained_demand"] = (
        pd.to_numeric(
            result["deconstrained_demand"],
            errors="coerce",
        )
        .fillna(result["quantity_sold"])
        .clip(lower=0.0)
    )

    result["estimated_lost_demand"] = (
        pd.to_numeric(
            result["estimated_lost_demand"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result["estimated_lost_demand"] = np.maximum(
        result["estimated_lost_demand"],
        result["deconstrained_demand"]
        - result["quantity_sold"],
    )

    result["demand_opportunity"] = (
        result["deconstrained_demand"]
    )

    result["lost_demand_rate"] = np.where(
        result["deconstrained_demand"] > 0,
        (
            result["estimated_lost_demand"]
            / result["deconstrained_demand"]
        ).clip(0.0, 1.0),
        0.0,
    )

    result["realized_demand_rate"] = np.where(
        result["deconstrained_demand"] > 0,
        (
            result["quantity_sold"]
            / result["deconstrained_demand"]
        ).clip(0.0, 1.0),
        1.0,
    )

    result["lost_demand_flag"] = (
        result["estimated_lost_demand"] > 0
    )

    result["material_lost_demand_flag"] = (
        result["lost_demand_rate"] >= 0.10
    )

    result["severe_lost_demand_flag"] = (
        result["lost_demand_rate"] >= 0.25
    )

    result["lost_demand_value_index"] = (
        result["estimated_lost_demand"]
        * result["lost_demand_rate"]
    )

    return result


def aggregate_lost_demand(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Aggregate lost-demand intelligence by any grouping.

    Typical groups:

        ["outlet_id"]
        ["product_id"]
        ["outlet_id", "product_id"]
    """

    validate_input(df)

    metrics = add_lost_demand_metrics(df)

    grouped = (
        metrics
        .groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            total_observed_sales=(
                "quantity_sold",
                "sum",
            ),
            total_deconstrained_demand=(
                "deconstrained_demand",
                "sum",
            ),
            total_estimated_lost_demand=(
                "estimated_lost_demand",
                "sum",
            ),
            average_lost_demand_rate=(
                "lost_demand_rate",
                "mean",
            ),
            stockout_days=(
                "stockout",
                "sum",
            ),
            lost_demand_days=(
                "lost_demand_flag",
                "sum",
            ),
            material_lost_demand_days=(
                "material_lost_demand_flag",
                "sum",
            ),
            severe_lost_demand_days=(
                "severe_lost_demand_flag",
                "sum",
            ),
            total_demand_opportunity=(
                "demand_opportunity",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["overall_lost_demand_rate"] = np.where(
        grouped["total_deconstrained_demand"] > 0,
        (
            grouped["total_estimated_lost_demand"]
            / grouped["total_deconstrained_demand"]
        ).clip(0.0, 1.0),
        0.0,
    )

    grouped["fulfillment_rate"] = np.where(
        grouped["total_deconstrained_demand"] > 0,
        (
            grouped["total_observed_sales"]
            / grouped["total_deconstrained_demand"]
        ).clip(0.0, 1.0),
        1.0,
    )

    return grouped