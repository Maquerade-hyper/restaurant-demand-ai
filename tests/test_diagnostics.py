import numpy as np
import pandas as pd

from app.forecasting.diagnostics import (
    calculate_errors,
    diagnostic_metrics,
    bias_analysis,
    high_demand_analysis,
    spike_recall,
    group_diagnostics,
)


def test_calculate_errors():

    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 35])

    result = calculate_errors(
        actual,
        predicted,
    )

    assert len(result) == 3
    assert result.iloc[0]["error"] == 2
    assert result.iloc[1]["absolute_error"] == 2


def test_diagnostic_metrics():

    actual = np.array([10, 20, 30])
    predicted = np.array([10, 20, 30])

    result = diagnostic_metrics(
        actual,
        predicted,
    )

    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert result["smape"] == 0
    assert result["bias"] == 0


def test_bias_analysis():

    actual = np.array([10, 10, 10])
    predicted = np.array([12, 8, 10])

    result = bias_analysis(
        actual,
        predicted,
    )

    assert result["mean_bias"] == 0
    assert result["overforecast_rate"] > 0
    assert result["underforecast_rate"] > 0


def test_high_demand_analysis():

    actual = np.arange(1, 101)
    predicted = actual.copy()

    result = high_demand_analysis(
        actual,
        predicted,
    )

    assert result["count"] > 0
    assert result["mae"] == 0


def test_spike_recall():

    actual = np.arange(1, 101)
    predicted = actual.copy()

    result = spike_recall(
        actual,
        predicted,
    )

    assert result == 1.0


def test_group_diagnostics():

    df = pd.DataFrame(
        {
            "outlet_id": [
                "O001",
                "O001",
                "O002",
                "O002",
            ],
            "quantity_sold": [
                10,
                20,
                30,
                40,
            ],
            "prediction": [
                10,
                20,
                30,
                40,
            ],
        }
    )

    result = group_diagnostics(
        df,
        group_column="outlet_id",
    )

    assert len(result) == 2
    assert result.iloc[0]["mae"] == 0