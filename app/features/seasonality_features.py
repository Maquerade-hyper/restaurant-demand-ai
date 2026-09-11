
from __future__ import annotations

import numpy as np
import pandas as pd


def create_seasonality_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["dow_sin_2"] = np.sin(
        4 * np.pi
        * result["day_of_week"]
        / 7
    )

    result["dow_cos_2"] = np.cos(
        4 * np.pi
        * result["day_of_week"]
        / 7
    )

    result["month_sin_2"] = np.sin(
        4 * np.pi
        * (result["month"] - 1)
        / 12
    )

    result["month_cos_2"] = np.cos(
        4 * np.pi
        * (result["month"] - 1)
        / 12
    )

    result["quarter_sin"] = np.sin(
        2 * np.pi
        * (result["quarter"] - 1)
        / 4
    )

    result["quarter_cos"] = np.cos(
        2 * np.pi
        * (result["quarter"] - 1)
        / 4
    )

    result["weekend_month_interaction"] = (
        result["is_weekend"]
        * result["month"]
    )

    return result

