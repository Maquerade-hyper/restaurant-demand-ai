from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "outlet_id",
    "product_id",
    "date",
    "quantity_sold",
    "closing_stock",
    "stockout",
}


def prepare_stockout_signals(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build availability and stockout signals from observed sales
    and inventory information.

    This function never uses true_demand or demand_truth.
    """

    sales_required = {
        "outlet_id",
        "product_id",
        "date",
        "quantity_sold",
    }

    inventory_required = {
        "outlet_id",
        "product_id",
        "date",
        "closing_stock",
        "stockout",
    }

    missing_sales = sales_required - set(sales.columns)
    missing_inventory = inventory_required - set(inventory.columns)

    if missing_sales:
        raise ValueError(
            f"Missing sales columns: {sorted(missing_sales)}"
        )

    if missing_inventory:
        raise ValueError(
            f"Missing inventory columns: {sorted(missing_inventory)}"
        )

    sales_df = sales.copy()
    inventory_df = inventory.copy()

    sales_df["date"] = pd.to_datetime(
        sales_df["date"]
    )

    inventory_df["date"] = pd.to_datetime(
        inventory_df["date"]
    )

    keys = [
        "outlet_id",
        "product_id",
        "date",
    ]

    inventory_columns = [
        "outlet_id",
        "product_id",
        "date",
        "closing_stock",
        "stockout",
    ]

    inventory_df = inventory_df[
        inventory_columns
    ].copy()

    result = sales_df.merge(
        inventory_df,
        on=keys,
        how="left",
    )

    result["closing_stock"] = (
        result["closing_stock"]
        .fillna(0.0)
        .astype(float)
    )

    result["stockout"] = (
        result["stockout"]
        .fillna(False)
        .astype(bool)
    )

    result["quantity_sold"] = (
        result["quantity_sold"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result = result.sort_values(
        ["outlet_id", "product_id", "date"]
    ).reset_index(drop=True)

    group = result.groupby(
        ["outlet_id", "product_id"],
        sort=False,
    )

    result["previous_stockout"] = (
        group["stockout"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result["stockout_start"] = (
        result["stockout"]
        & ~result["previous_stockout"]
    )

    result["stockout_duration"] = (
        result.groupby(
            ["outlet_id", "product_id"]
        )["stockout"]
        .transform(
            lambda x: (
                x.astype(int)
                .groupby(
                    (~x).cumsum()
                )
                .cumsum()
            )
        )
    )

    result["post_stockout_recovery"] = (
        ~result["stockout"]
        & result["previous_stockout"]
    )

    result["available"] = (
        ~result["stockout"]
    ).astype(int)

    return result