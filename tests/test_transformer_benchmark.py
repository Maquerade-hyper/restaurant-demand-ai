from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.forecasting.transformer.dataset import (
    TemporalSequenceDataset,
)

from app.forecasting.transformer.model import (
    TemporalTransformer,
)

from app.forecasting.transformer.service import (
    TransformerBenchmarkService,
)


def build_frame():

    rng = np.random.default_rng(
        42
    )

    dates = pd.date_range(
        "2025-01-01",
        periods=180,
        freq="D",
    )

    rows = []

    for outlet in [
        "O001",
        "O002",
        "O003",
    ]:

        for product in [
            "P001",
            "P002",
        ]:

            base = (
                50
                +
                rng.normal(
                    0,
                    2,
                )
            )

            values = (
                base
                +
                5
                *
                np.sin(
                    np.arange(180)
                    *
                    2
                    *
                    np.pi
                    /
                    7
                )
                +
                rng.normal(
                    0,
                    1.5,
                    180,
                )
            )

            for date, value in zip(
                dates,
                values,
            ):

                rows.append(
                    {
                        "outlet_id": outlet,
                        "product_id": product,
                        "date": date,
                        "deconstrained_demand": max(
                            0,
                            float(value),
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def test_temporal_dataset():

    values = np.arange(
        50,
        dtype=np.float32,
    )

    dataset = TemporalSequenceDataset(
        values,
        sequence_length=14,
    )

    assert len(dataset) == 36

    x, y = dataset[0]

    assert x.shape == (
        14,
        1,
    )

    assert y.ndim == 0


def test_transformer_forward():

    model = TemporalTransformer(
        input_size=1,
        d_model=32,
        n_heads=4,
        n_layers=2,
        feedforward_dim=64,
        max_sequence_length=14,
    )

    x = torch.randn(
        4,
        14,
        1,
    )

    output = model(x)

    assert output.shape == (
        4,
    )


def test_multiseries_benchmark():

    frame = build_frame()

    service = TransformerBenchmarkService(
        sequence_length=14,
        test_size=14,
        validation_size=14,
        epochs=2,
        batch_size=16,
    )

    aggregate, detail = (
        service.benchmark(
            frame,
            max_series=3,
        )
    )

    assert set(
        aggregate["model_name"]
    ) == {
        "naive",
        "xgboost",
        "transformer",
    }

    assert len(detail) == 3


def test_benchmark_metrics_finite():

    frame = build_frame()

    service = TransformerBenchmarkService(
        sequence_length=14,
        test_size=14,
        validation_size=14,
        epochs=2,
        batch_size=16,
    )

    aggregate, detail = (
        service.benchmark(
            frame,
            max_series=3,
        )
    )

    metric_columns = [
        "mae",
        "rmse",
        "smape",
        "bias",
        "high_demand_mae",
        "high_demand_bias",
        "spike_recall",
        "prediction_seconds",
        "prediction_cost",
    ]

    for column in metric_columns:

        assert np.isfinite(
            aggregate[column].to_numpy()
        ).all()


def test_validation():

    frame = build_frame()

    service = TransformerBenchmarkService(
        sequence_length=14,
        test_size=14,
        validation_size=14,
        epochs=2,
        batch_size=16,
    )

    aggregate, detail = (
        service.benchmark(
            frame,
            max_series=3,
        )
    )

    result = service.validate(
        aggregate,
        detail,
    )

    assert result["passed"] is True

    assert result["errors"] == []


def test_production_decision():

    frame = build_frame()

    service = TransformerBenchmarkService(
        sequence_length=14,
        test_size=14,
        validation_size=14,
        epochs=2,
        batch_size=16,
    )

    aggregate, detail = (
        service.benchmark(
            frame,
            max_series=3,
        )
    )

    decision = (
        service.production_decision(
            aggregate,
            detail,
        )
    )

    assert decision[
        "recommendation"
    ] in {
        "KEEP_XGBOOST",
        "TRANSFORMER_CANDIDATE",
    }

    assert np.isfinite(
        decision[
            "transformer_mae_improvement"
        ]
    )


def test_transformer_candidate_requires_both_conditions():

    service = TransformerBenchmarkService(
        improvement_threshold=0.02,
        minimum_series_win_rate=0.50,
    )

    aggregate = pd.DataFrame(
        [
            {
                "model_name": "naive",
                "mae": 12,
                "rmse": 13,
                "smape": 20,
                "bias": 0,
                "high_demand_mae": 15,
                "high_demand_bias": 0,
                "spike_recall": 0.5,
                "prediction_seconds": 0,
                "prediction_cost": 0,
                "rank": 1,
            },
            {
                "model_name": "xgboost",
                "mae": 10,
                "rmse": 11,
                "smape": 15,
                "bias": 0,
                "high_demand_mae": 12,
                "high_demand_bias": 0,
                "spike_recall": 0.7,
                "prediction_seconds": 1,
                "prediction_cost": 0.01,
                "rank": 2,
            },
            {
                "model_name": "transformer",
                "mae": 9.5,
                "rmse": 10,
                "smape": 14,
                "bias": 0,
                "high_demand_mae": 11,
                "high_demand_bias": 0,
                "spike_recall": 0.75,
                "prediction_seconds": 2,
                "prediction_cost": 0.02,
                "rank": 3,
            },
        ]
    )

    detail = pd.DataFrame(
        [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "naive_mae": 12,
                "xgboost_mae": 10,
                "transformer_mae": 9,
                "best_model": "transformer",
            },
            {
                "outlet_id": "O002",
                "product_id": "P001",
                "naive_mae": 12,
                "xgboost_mae": 10,
                "transformer_mae": 11,
                "best_model": "xgboost",
            },
        ]
    )

    decision = (
        service.production_decision(
            aggregate,
            detail,
        )
    )

    assert (
        decision[
            "recommendation"
        ]
        ==
        "TRANSFORMER_CANDIDATE"
    )