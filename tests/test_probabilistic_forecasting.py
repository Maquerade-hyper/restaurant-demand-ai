from __future__ import annotations

import numpy as np

from app.forecasting.probabilistic.service import (
    ProbabilisticForecastingService,
)

from app.forecasting.probabilistic.conformal import (
    conformal_quantile,
)


def calibration_data():

    rng = np.random.default_rng(42)

    prediction = np.full(
        200,
        50.0,
    )

    residual = rng.normal(
        0.0,
        5.0,
        200,
    )

    actual = (
        prediction
        +
        residual
    )

    return actual, prediction


# ============================================================
# CONFORMAL
# ============================================================

def test_conformal_radius_is_positive():

    actual, prediction = calibration_data()

    radius = conformal_quantile(
        actual,
        prediction,
        coverage=0.90,
    )

    assert radius >= 0.0


# ============================================================
# QUANTILES
# ============================================================

def test_quantile_forecasts_exist():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.quantiles(
        np.array(
            [50.0, 55.0, 60.0]
        ),
        actual,
        prediction,
    )

    assert {
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
    }.issubset(
        result.keys()
    )


# ============================================================
# QUANTILE ORDER
# ============================================================

def test_quantiles_are_monotonic():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.quantiles(
        np.array(
            [50.0, 55.0, 60.0]
        ),
        actual,
        prediction,
    )

    assert np.all(
        result["p10"]
        <=
        result["p25"]
    )

    assert np.all(
        result["p25"]
        <=
        result["p50"]
    )

    assert np.all(
        result["p50"]
        <=
        result["p75"]
    )

    assert np.all(
        result["p75"]
        <=
        result["p90"]
    )


# ============================================================
# INTERVAL
# ============================================================

def test_conformal_interval_is_valid():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.interval(
        np.array(
            [40.0, 50.0, 60.0]
        ),
        actual,
        prediction,
    )

    assert np.all(
        result["lower"]
        <=
        result["upper"]
    )

    assert result["radius"] >= 0.0


# ============================================================
# NON-NEGATIVE DEMAND
# ============================================================

def test_probabilistic_forecast_is_non_negative():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.forecast(
        np.array(
            [1.0, 2.0, 3.0]
        ),
        actual,
        prediction,
    )

    for values in result[
        "quantiles"
    ].values():

        assert np.all(
            values >= 0.0
        )

    assert np.all(
        result["interval"]["lower"]
        >=
        0.0
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def test_predictive_distribution_exists():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.distribution(
        np.array(
            [50.0, 60.0]
        ),
        actual,
        prediction,
    )

    expected = {
        "point_forecast",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "predictive_std",
        "predictive_width_80",
        "predictive_width_90",
    }

    assert expected.issubset(
        result.columns
    )


# ============================================================
# FULL FORECAST
# ============================================================

def test_full_forecast_contract():

    actual, prediction = calibration_data()

    service = ProbabilisticForecastingService()

    result = service.forecast(
        np.array(
            [40.0, 50.0, 60.0]
        ),
        actual,
        prediction,
    )

    validation = (
        service.validate_forecast(
            result
        )
    )

    assert validation["passed"] is True

    assert validation["errors"] == []


# ============================================================
# CALIBRATION EVALUATION
# ============================================================

def test_calibration_evaluation():

    actual, prediction = calibration_data()

    point = np.full(
        50,
        50.0,
    )

    service = ProbabilisticForecastingService()

    evaluation = service.evaluate(
        actual[:50],
        point,
        actual[50:],
        prediction[50:],
    )

    assert (
        "quantiles"
        in evaluation
    )

    assert (
        "intervals"
        in evaluation
    )

    assert (
        "p50"
        in evaluation["quantiles"]
    )

    assert (
        "conformal_90"
        in evaluation["intervals"]
    )


# ============================================================
# VALIDATION REJECTS QUANTILE CROSSING
# ============================================================

def test_validation_rejects_quantile_crossing():

    service = ProbabilisticForecastingService()

    forecast = {
        "quantiles": {
            "p10": np.array([10.0]),
            "p25": np.array([20.0]),
            "p50": np.array([15.0]),
            "p75": np.array([30.0]),
            "p90": np.array([40.0]),
        },
        "interval": {
            "lower": np.array([10.0]),
            "upper": np.array([40.0]),
        },
        "distribution": object(),
    }

    validation = (
        service.validate_forecast(
            forecast
        )
    )

    assert validation["passed"] is False