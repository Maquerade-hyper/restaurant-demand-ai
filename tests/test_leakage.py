import numpy as np
import pandas as pd

from app.forecasting.leakage import (
    audit_feature_columns,
    audit_temporal_split,
)
from app.forecasting.feature_audit import (
    audit_lag_features,
)


def test_forbidden_feature_detection():

    result = audit_feature_columns(
        [
            "lag_1",
            "rolling_mean_7",
            "quantity_sold",
        ]
    )

    assert result["passed"] is False

    assert (
        "quantity_sold"
        in result["violations"]
    )


def test_clean_feature_columns():

    result = audit_feature_columns(
        [
            "lag_1",
            "rolling_mean_7",
            "day_of_week",
        ]
    )

    assert result["passed"] is True


def test_temporal_split():

    train = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=10,
            )
        }
    )

    test = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-11",
                periods=5,
            )
        }
    )

    result = audit_temporal_split(
        train,
        test,
    )

    assert result["passed"] is True


def test_lag_features():

    df = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O001",
                "O001",
                "O001",
            ],
            "product_id": [
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "date": pd.date_range(
                "2025-01-01",
                periods=4,
            ),
            "quantity_sold": [
                10,
                20,
                30,
                40,
            ],
        }
    )

    df["lag_1"] = (
        df["quantity_sold"]
        .shift(1)
    )

    result = audit_lag_features(
        df
    )

    assert result["passed"] is True