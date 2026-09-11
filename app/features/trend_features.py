
from __future__ import annotations

import numpy as np
import pandas as pd


def create_trend_features(
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

    lag_1 = grouped[target_column].shift(1)
    lag_7 = grouped[target_column].shift(7)
    lag_14 = grouped[target_column].shift(14)

    result["trend_1d"] = (
        lag_1
        - grouped[target_column].shift(2)
    )

    result["trend_7d"] = (
        lag_1 - lag_7
    )

    result["trend_14d"] = (
        lag_1 - lag_14
    )

    result["trend_ratio_7d"] = (
        lag_1
        / lag_7.replace(0, np.nan)
    )

    result["trend_ratio_14d"] = (
        lag_1
        / lag_14.replace(0, np.nan)
    )

    result["trend_acceleration"] = (
        result["trend_7d"]
        - result["trend_14d"]
    )

    return result
