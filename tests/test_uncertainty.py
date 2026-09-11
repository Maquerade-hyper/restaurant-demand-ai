import numpy as np
import pandas as pd

from app.forecasting.uncertainty.error_profiles import (
    calculate_error_profiles,
)

from app.forecasting.uncertainty.prediction_uncertainty import (
    EmpiricalPredictionInterval,
)

from app.forecasting.uncertainty.reliability import (
    interval_coverage,
)

from app.forecasting.uncertainty.confidence import (
    calculate_confidence,
)


def test_error_profiles():

    actual = pd.Series(
        [10, 12, 15, 20, 25]
    )

    prediction = pd.Series(
        [9, 13, 14, 18, 27]
    )

    result = calculate_error_profiles(
        actual,
        prediction,
    )

    assert result["count"] == 5
    assert result["mae"] >= 0
    assert result["p90_absolute_error"] >= 0


def test_prediction_interval():

    actual = pd.Series(
        [10, 12, 15, 20, 25]
    )

    prediction = pd.Series(
        [9, 13, 14, 18, 27]
    )

    model = EmpiricalPredictionInterval()

    model.fit(
        actual,
        prediction,
    )

    result = model.predict_interval(
        [10, 20, 30]
    )

    assert "prediction" in result
    assert "lower_bound" in result
    assert "upper_bound" in result

    assert (
        result["lower_bound"]
        <= result["upper_bound"]
    ).all()


def test_interval_coverage():

    actual = pd.Series(
        [10, 20, 30]
    )

    lower = pd.Series(
        [5, 15, 25]
    )

    upper = pd.Series(
        [15, 25, 35]
    )

    coverage = interval_coverage(
        actual,
        lower,
        upper,
    )

    assert coverage == 1.0


def test_confidence():

    confidence = calculate_confidence(
        prediction=[100, 100],
        lower_bound=[90, 50],
        upper_bound=[110, 150],
    )

    assert len(confidence) == 2
    assert confidence[0] > confidence[1]
    assert np.all(
        (confidence >= 0)
        & (confidence <= 1)
    )