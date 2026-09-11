
from __future__ import annotations

import numpy as np
import pandas as pd


def create_volatility_features(
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

    for window in [3, 7, 14, 28]:

        rolling_mean = (
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

        rolling_std = (
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
                ).std()
            )
        )

        result[
            f"volatility_{window}"
        ] = rolling_std

        result[
            f"coefficient_variation_{window}"
        ] = (
            rolling_std
            / rolling_mean.replace(
                0,
                np.nan,
            )
        )

    return result

