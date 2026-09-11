import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
    group_columns: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:

    result = df.copy()

    group_columns = group_columns or [
        "outlet_id",
        "product_id",
    ]

    lags = lags or [
        1,
        2,
        3,
        7,
        14,
        28,
    ]

    result = result.sort_values(
        group_columns + ["date"]
    )

    grouped = result.groupby(
        group_columns,
        sort=False,
    )[target_column]

    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(lag)

    return result