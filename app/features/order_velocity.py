
from __future__ import annotations

import pandas as pd


def create_order_velocity_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:

    if group_columns is None:
        group_columns = [
            "outlet_id",
            "product_id",
        ]

    result = df.copy()

    result = result.sort_values(
        group_columns + ["date"]
    ).reset_index(drop=True)

    grouped = result.groupby(
        group_columns,
        sort=False,
    )

    shifted = grouped[
        target_column
    ].shift(1)

    for window in [3, 7, 14]:

        velocity = (
            shifted
            .groupby(
                [
                    result["outlet_id"],
                    result["product_id"],
                ]
            )
            .transform(
                lambda x: x.rolling(
                    window
                ).mean()
            )
        )

        result[
            f"order_velocity_{window}d"
        ] = velocity

    result[
        "order_velocity_change"
    ] = (
        result["order_velocity_3d"]
        - result["order_velocity_14d"]
    )

    return result

