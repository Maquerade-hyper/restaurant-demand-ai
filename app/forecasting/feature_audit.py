from __future__ import annotations

import numpy as np
import pandas as pd


def audit_lag_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
) -> dict:

    ordered = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    lag_columns = [
        column
        for column in ordered.columns
        if column.startswith("lag_")
    ]

    violations = []

    for column in lag_columns:

        lag_number = int(
            column.split("_")[1]
        )

        expected = (
            ordered
            .groupby(
                [
                    "outlet_id",
                    "product_id",
                ]
            )[target_column]
            .shift(lag_number)
        )

        actual = ordered[column]

        if not np.allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            equal_nan=True,
        ):
            violations.append(column)

    return {
        "passed": len(violations) == 0,
        "lag_columns": lag_columns,
        "violations": violations,
    }


def audit_rolling_features(
    df: pd.DataFrame,
    target_column: str = "quantity_sold",
) -> dict:

    ordered = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    rolling_columns = [
        column
        for column in ordered.columns
        if column.startswith("rolling_mean_")
        or column.startswith("rolling_std_")
    ]

    violations = []

    grouped = ordered.groupby(
        [
            "outlet_id",
            "product_id",
        ],
        sort=False,
    )

    shifted = grouped[
        target_column
    ].shift(1)

    for column in rolling_columns:

        window = int(
            column.split("_")[-1]
        )

        if column.startswith(
            "rolling_mean_"
        ):

            expected = (
                shifted
                .groupby(
                    [
                        ordered["outlet_id"],
                        ordered["product_id"],
                    ]
                )
                .transform(
                    lambda x: x.rolling(
                        window
                    ).mean()
                )
            )

        else:

            expected = (
                shifted
                .groupby(
                    [
                        ordered["outlet_id"],
                        ordered["product_id"],
                    ]
                )
                .transform(
                    lambda x: x.rolling(
                        window
                    ).std()
                )
            )

        actual = ordered[column]

        if not np.allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            equal_nan=True,
        ):
            violations.append(column)

    return {
        "passed": len(violations) == 0,
        "rolling_columns": rolling_columns,
        "violations": violations,
    }