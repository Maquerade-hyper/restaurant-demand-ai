from __future__ import annotations

import numpy as np
import pandas as pd


def add_outlet_product_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
) -> pd.DataFrame:
    """
    Part 22C - Outlet × Product Intelligence.

    All demand statistics are historical.

    No current or future target value is used to construct
    the current-row features.
    """

    required = {
        "date",
        "outlet_id",
        "product_id",
        target_column,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        return df.copy()

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"]
    )

    # Preserve original order.
    out["_advanced_original_order"] = np.arange(
        len(out)
    )

    # Chronological order.
    out = (
        out.sort_values(
            [
                "outlet_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    pair_keys = [
        "outlet_id",
        "product_id",
    ]

    eps = 1e-6

    # ============================================================
    # HISTORICAL PAIR DEMAND
    # ============================================================

    out["_advanced_pair_history"] = (
        out.groupby(
            pair_keys,
            sort=False,
        )[target_column]
        .shift(1)
        .astype(float)
    )

    history = out["_advanced_pair_history"]

    # ============================================================
    # PAIR EXPANDING STATISTICS
    # ============================================================

    pair_group = out.groupby(
        pair_keys,
        sort=False,
    )["_advanced_pair_history"]

    out["advanced_pair_mean"] = (
        pair_group
        .expanding(
            min_periods=1
        )
        .mean()
        .reset_index(
            level=pair_keys,
            drop=True,
        )
    )

    out["advanced_pair_std"] = (
        pair_group
        .expanding(
            min_periods=2
        )
        .std()
        .reset_index(
            level=pair_keys,
            drop=True,
        )
    )

    out["advanced_pair_cv"] = (
        out["advanced_pair_std"]
        / (
            out["advanced_pair_mean"].abs()
            + eps
        )
    )

    # ============================================================
    # DAILY OUTLET HISTORICAL DEMAND
    # ============================================================

    daily_outlet = (
        out.groupby(
            [
                "outlet_id",
                "date",
            ],
            as_index=False,
        )["_advanced_pair_history"]
        .sum()
        .rename(
            columns={
                "_advanced_pair_history":
                "_outlet_history"
            }
        )
    )

    daily_outlet = (
        daily_outlet
        .sort_values(
            [
                "outlet_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    outlet_group = daily_outlet.groupby(
        "outlet_id",
        sort=False,
    )["_outlet_history"]

    daily_outlet["_outlet_mean"] = (
        outlet_group
        .expanding(
            min_periods=1
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    daily_outlet["_outlet_std"] = (
        outlet_group
        .expanding(
            min_periods=2
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    out = out.merge(
        daily_outlet[
            [
                "outlet_id",
                "date",
                "_outlet_mean",
                "_outlet_std",
            ]
        ],
        on=[
            "outlet_id",
            "date",
        ],
        how="left",
    )

    out["advanced_outlet_mean"] = (
        out["_outlet_mean"]
    )

    # Required public contract name.
    out["advanced_outlet_mean_demand"] = (
        out["advanced_outlet_mean"]
    )

    out["advanced_outlet_std"] = (
        out["_outlet_std"]
    )

    # ============================================================
    # DAILY PRODUCT HISTORICAL DEMAND
    # ============================================================

    daily_product = (
        out.groupby(
            [
                "product_id",
                "date",
            ],
            as_index=False,
        )["_advanced_pair_history"]
        .sum()
        .rename(
            columns={
                "_advanced_pair_history":
                "_product_history"
            }
        )
    )

    daily_product = (
        daily_product
        .sort_values(
            [
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    product_group = daily_product.groupby(
        "product_id",
        sort=False,
    )["_product_history"]

    daily_product["_product_mean"] = (
        product_group
        .expanding(
            min_periods=1
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    daily_product["_product_std"] = (
        product_group
        .expanding(
            min_periods=2
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    out = out.merge(
        daily_product[
            [
                "product_id",
                "date",
                "_product_mean",
                "_product_std",
            ]
        ],
        on=[
            "product_id",
            "date",
        ],
        how="left",
    )

    out["advanced_product_mean"] = (
        out["_product_mean"]
    )

    # Required public contract name.
    out["advanced_product_mean_demand"] = (
        out["advanced_product_mean"]
    )

    out["advanced_product_std"] = (
        out["_product_std"]
    )

    # ============================================================
    # RELATIVE DEMAND
    # ============================================================

    out["advanced_outlet_relative_demand"] = (
        out["advanced_pair_mean"]
        / (
            out["advanced_outlet_mean"].abs()
            + eps
        )
    )

    out["advanced_product_relative_demand"] = (
        out["advanced_pair_mean"]
        / (
            out["advanced_product_mean"].abs()
            + eps
        )
    )

    # ============================================================
    # HISTORICAL PRODUCT SHARE OF OUTLET
    # ============================================================

    outlet_total = (
        out.groupby(
            [
                "outlet_id",
                "date",
            ],
            sort=False,
        )["_advanced_pair_history"]
        .transform("sum")
    )

    out["advanced_product_share_of_outlet"] = (
        history
        / (
            outlet_total.abs()
            + eps
        )
    )

    # ============================================================
    # HISTORICAL OUTLET SHARE OF PRODUCT
    # ============================================================

    product_total = (
        out.groupby(
            [
                "product_id",
                "date",
            ],
            sort=False,
        )["_advanced_pair_history"]
        .transform("sum")
    )

    out["advanced_outlet_share_of_product"] = (
        history
        / (
            product_total.abs()
            + eps
        )
    )

    # ============================================================
    # RELATIVE PAIR POSITION
    # ============================================================

    out["advanced_pair_vs_outlet"] = (
        out["advanced_pair_mean"]
        - out["advanced_outlet_mean"]
    )

    out["advanced_pair_vs_product"] = (
        out["advanced_pair_mean"]
        - out["advanced_product_mean"]
    )

    out["advanced_pair_intensity"] = (
        out["advanced_pair_mean"]
        / (
            out["advanced_product_mean"].abs()
            + eps
        )
    )

    # ============================================================
    # OUTLET × PRODUCT AFFINITY
    # ============================================================

    #
    # Measures the historical pair demand relative to the
    # expected combination of outlet-level and product-level
    # historical demand.
    #
    # > 1  = stronger-than-expected affinity
    # < 1  = weaker-than-expected affinity
    #

    denominator = (
        out["advanced_outlet_mean"]
        *
        out["advanced_product_mean"]
    )

    out["advanced_outlet_product_affinity"] = (
        out["advanced_pair_mean"]
        /
        (
            denominator.abs()
            + eps
        )
    )

    # ============================================================
    # CLEAN NUMERICAL VALUES
    # ============================================================

    numeric_columns = out.select_dtypes(
        include=[np.number]
    ).columns

    out[numeric_columns] = (
        out[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # ============================================================
    # REMOVE INTERNAL COLUMNS
    # ============================================================

    out.drop(
        columns=[
            "_advanced_pair_history",
            "_outlet_mean",
            "_outlet_std",
            "_product_mean",
            "_product_std",
            "_outlet_history",
            "_product_history",
        ],
        inplace=True,
        errors="ignore",
    )

    # ============================================================
    # RESTORE ORIGINAL ORDER
    # ============================================================

    out = (
        out
        .sort_values(
            "_advanced_original_order"
        )
        .drop(
            columns=[
                "_advanced_original_order",
            ]
        )
        .reset_index(drop=True)
    )

    return out