from __future__ import annotations

import pandas as pd

from app.features.feature_pipeline import build_features


def prepare_xgboost_dataset(
    sales: pd.DataFrame,
) -> pd.DataFrame:

    df = sales.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["outlet_id", "product_id", "date"]
    ).reset_index(drop=True)

    df = build_features(
        df,
        date_column="date",
        target_column="quantity_sold",
        group_columns=[
            "outlet_id",
            "product_id",
        ],
    )

    return df