import pandas as pd

from app.forecasting.baseline import (
    baseline_forecast,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from app.forecasting.metrics import (
    evaluate_forecast,
)


def test_naive():

    history = pd.Series(
        [10, 20, 30]
    )

    result = naive_forecast(
        history,
        3,
    )

    assert result.tolist() == [
        30.0,
        30.0,
        30.0,
    ]


def test_moving_average():

    history = pd.Series(
        [10, 20, 30]
    )

    result = moving_average_forecast(
        history,
        2,
        window=2,
    )

    assert result.tolist() == [
        25.0,
        25.0,
    ]


def test_seasonal_naive():

    history = pd.Series(
        [1, 2, 3, 4, 5, 6, 7]
    )

    result = seasonal_naive_forecast(
        history,
        14,
        season_length=7,
    )

    assert result.tolist() == [
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6, 7,
    ]


def test_baseline_dispatch():

    history = pd.Series(
        [10, 20, 30]
    )

    result = baseline_forecast(
        history,
        2,
        method="naive",
    )

    assert len(result) == 2


def test_metrics():

    actual = [10, 20, 30]
    predicted = [10, 25, 25]

    result = evaluate_forecast(
        actual,
        predicted,
    )

    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["smape"] >= 0