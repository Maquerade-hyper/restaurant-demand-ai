import pandas as pd

from app.forecasting.multihorizon.forecast import (
    MultiHorizonForecastService,
)


SALES_PATH = "data/synthetic/sales.csv"


def test_d7_recursive_forecast():

    sales = pd.read_csv(
        SALES_PATH
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    history = sales[
        sales["date"]
        <= pd.Timestamp("2025-12-02")
    ].copy()

    service = MultiHorizonForecastService()

    service.train(
        sales=history,
        cutoff_date=pd.Timestamp(
            "2025-12-03"
        ),
    )

    result = service.forecast_series(
        history=history,
        outlet_id="O001",
        product_id="P001",
        start_date=pd.Timestamp(
            "2025-12-03"
        ),
        horizon=7,
    )

    assert len(result) == 7

    assert list(
        result["horizon"]
    ) == [1, 2, 3, 4, 5, 6, 7]

    expected_dates = pd.date_range(
        "2025-12-03",
        "2025-12-09",
        freq="D",
    )

    assert list(
        result["date"]
    ) == list(expected_dates)

    assert (
        result["prediction"] >= 0
    ).all()

    assert (
        result["prediction"].notna()
    ).all()