import pandas as pd

from app.forecasting.multihorizon.evaluation import (
    evaluate_horizons,
)


def test_evaluate_horizons():

    actual = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O001",
                "O001",
            ],
            "product_id": [
                "P001",
                "P001",
                "P001",
            ],
            "date": pd.date_range(
                "2025-12-03",
                periods=3,
            ),
            "quantity_sold": [
                50.0,
                60.0,
                70.0,
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O001",
                "O001",
            ],
            "product_id": [
                "P001",
                "P001",
                "P001",
            ],
            "date": pd.date_range(
                "2025-12-03",
                periods=3,
            ),
            "horizon": [
                1,
                2,
                3,
            ],
            "prediction": [
                52.0,
                58.0,
                73.0,
            ],
        }
    )

    result = evaluate_horizons(
        actual,
        predictions,
    )

    assert len(result) == 3

    assert list(
        result["horizon"]
    ) == [1, 2, 3]

    assert (
        result["mae"] >= 0
    ).all()