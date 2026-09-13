import numpy as np
import pandas as pd
import pytest

from app.forecasting.benchmarking import (
    ModelBenchmarkingService,
)


def build_series():

    rng = np.random.default_rng(
        42
    )

    history = (
        50
        + np.arange(84) * 0.15
        + 8
        * np.sin(
            np.arange(84)
            * 2
            * np.pi
            / 7
        )
        + rng.normal(
            0,
            1,
            84,
        )
    )

    test = (
        50
        + np.arange(14, dtype=float)
        * 0.15
        + 8
        * np.sin(
            np.arange(84, 98)
            * 2
            * np.pi
            / 7
        )
        + rng.normal(
            0,
            1,
            14,
        )
    )

    return (
        pd.Series(history),
        pd.Series(test),
    )


def test_complete_model_benchmarking():

    history, test = (
        build_series()
    )

    service = (
        ModelBenchmarkingService()
    )

    result = service.analyze(
        history=history,
        test=test,
    )

    assert not result.empty

    assert set(
        [
            "naive",
            "seasonal_naive",
            "moving_average",
        ]
    ).issubset(
        set(result["model_name"])
    )

    required = {
        "model_name",
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "high_demand_bias",
        "spike_recall",
        "stability",
        "prediction_cost",
        "score",
        "rank",
    }

    assert required.issubset(
        result.columns
    )


def test_metrics_are_valid():

    history, test = (
        build_series()
    )

    service = (
        ModelBenchmarkingService()
    )

    result = service.analyze(
        history=history,
        test=test,
    )

    assert (
        result["mae"] >= 0
    ).all()

    assert (
        result["rmse"] >= 0
    ).all()

    assert (
        result["smape"] >= 0
    ).all()

    assert (
        result["spike_recall"]
        .between(0, 1)
        .all()
    )

    assert (
        result["stability"]
        .between(0, 1)
        .all()
    )


def test_ranking_is_unique():

    history, test = (
        build_series()
    )

    service = (
        ModelBenchmarkingService()
    )

    result = service.analyze(
        history=history,
        test=test,
    )

    assert (
        result["rank"]
        .is_unique
    )

    assert (
        result["rank"].min()
        == 1
    )


def test_xgboost_result_can_be_benchmarked():

    history, test = (
        build_series()
    )

    service = (
        ModelBenchmarkingService()
    )

    fake_xgb = (
        test.to_numpy()
        + 0.5
    )

    result = service.analyze(
        history=history,
        test=test,
        xgb_actual=test,
        xgb_prediction=pd.Series(
            fake_xgb
        ),
        xgb_prediction_cost=0.05,
    )

    assert (
        "xgboost"
        in set(result["model_name"])
    )

    xgb = result[
        result["model_name"]
        == "xgboost"
    ].iloc[0]

    assert xgb["mae"] >= 0


def test_truth_columns_are_not_part_of_contract():

    history, test = (
        build_series()
    )

    service = (
        ModelBenchmarkingService()
    )

    result = service.analyze(
        history=history,
        test=test,
    )

    forbidden = {
        "true_demand",
        "lost_demand_truth",
        "demand_truth",
        "future_demand",
    }

    assert not (
        forbidden
        & set(result.columns)
    )


def test_empty_history_is_rejected():

    service = (
        ModelBenchmarkingService()
    )

    with pytest.raises(ValueError):

        service.analyze(
            history=pd.Series(
                dtype=float
            ),
            test=pd.Series(
                [1.0, 2.0]
            ),
        )