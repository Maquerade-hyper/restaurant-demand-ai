import pandas as pd


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:

    result = df.copy()

    group_columns = group_columns or [
        "outlet_id",
        "product_id",
    ]

    result = result.sort_values(
        group_columns + ["date"]
    )

    grouped = result.groupby(
        group_columns,
        sort=False,
    )[target_column]

    # Shift first so today's target never enters
    # today's rolling statistics.
    historical = grouped.shift(1)

    for window in [3, 7, 14, 28]:

        result[f"rolling_mean_{window}"] = (
            historical
            .groupby(
                [
                    result[column]
                    for column in group_columns
                ]
            )
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        result[f"rolling_std_{window}"] = (
            historical
            .groupby(
                [
                    result[column]
                    for column in group_columns
                ]
            )
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

    return result