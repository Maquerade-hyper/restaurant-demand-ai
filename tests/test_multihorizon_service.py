import pandas as pd

from app.forecasting.multihorizon.service import (
    MultiHorizonService,
)


def test_all_horizons():

    sales = pd.read_csv(
        "data/synthetic/sales.csv"
    )

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    history = sales[
        sales["date"]
        <= pd.Timestamp("2025-12-02")
    ].copy()

    service = MultiHorizonService()

    service.train(
        sales=history,
        cutoff_date=pd.Timestamp(
            "2025-12-03"
        ),
    )

    forecasts = (
        service.forecast_all_horizons(
            history=history,
            start_date=pd.Timestamp(
                "2025-12-03"
            ),
        )
    )

    assert set(
        forecasts.keys()
    ) == {1, 3, 7}

    assert len(
        forecasts[1]
    ) == 2600

    assert len(
        forecasts[3]
    ) == 2600 * 3

    assert len(
        forecasts[7]
    ) == 2600 * 7

    assert (
        forecasts[1]["horizon"]
        .max()
        == 1
    )

    assert (
        forecasts[3]["horizon"]
        .max()
        == 3
    )

    assert (
        forecasts[7]["horizon"]
        .max()
        == 7
    )

    for dataframe in forecasts.values():

        assert (
            dataframe["prediction"]
            >= 0
        ).all()

        assert (
            dataframe["prediction"]
            .notna()
        ).all()