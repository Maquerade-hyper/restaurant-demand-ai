from __future__ import annotations

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


def validate_event_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def build_stockout_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert daily stockout observations into contiguous stockout events.

    An event is a consecutive run of stockout days for the same
    outlet/product pair.
    """

    validate_event_input(df)

    result = df.copy()

    result["date"] = pd.to_datetime(result["date"])

    result["stockout"] = (
        result["stockout"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result = result.sort_values(
        ["outlet_id", "product_id", "date"]
    ).reset_index(drop=True)

    groups = ["outlet_id", "product_id"]

    previous_stockout = (
        result.groupby(groups)["stockout"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    event_start = result["stockout"] & ~previous_stockout

    result["stockout_event_start"] = event_start

    result["stockout_event_id"] = (
        event_start
        .groupby(
            [
                result["outlet_id"],
                result["product_id"],
            ]
        )
        .cumsum()
    )

    result["stockout_event_id"] = (
        result["stockout_event_id"]
        .where(result["stockout"], 0)
        .astype(int)
    )

    result["stockout_event_active"] = (
        result["stockout_event_id"] > 0
    )

    result["stockout_event_day"] = (
        result.groupby(
            [
                "outlet_id",
                "product_id",
                "stockout_event_id",
            ]
        )
        .cumcount()
        .add(1)
        .where(result["stockout"], 0)
    )

    return result


def summarize_stockout_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one record per contiguous stockout event.
    """

    result = build_stockout_events(df)

    events = result[result["stockout"]].copy()

    if events.empty:
        return pd.DataFrame(
            columns=[
                "outlet_id",
                "product_id",
                "stockout_event_id",
                "event_start",
                "event_end",
                "event_duration_days",
                "event_lost_demand",
                "event_demand",
                "event_sales",
            ]
        )

    summary = (
        events.groupby(
            [
                "outlet_id",
                "product_id",
                "stockout_event_id",
            ],
            as_index=False,
        )
        .agg(
            event_start=("date", "min"),
            event_end=("date", "max"),
            event_duration_days=("date", "count"),
            event_lost_demand=(
                "estimated_lost_demand",
                "sum",
            ),
            event_demand=(
                "deconstrained_demand",
                "sum",
            ),
            event_sales=(
                "quantity_sold",
                "sum",
            ),
        )
    )

    summary["event_loss_rate"] = (
        summary["event_lost_demand"]
        / summary["event_demand"].clip(lower=1e-9)
    ).clip(0.0, 1.0)

    summary["event_fulfillment_rate"] = (
        summary["event_sales"]
        / summary["event_demand"].clip(lower=1e-9)
    ).clip(0.0, 1.0)

    return summary