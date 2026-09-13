import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from app.production import (
    BatchInferenceEngine,
    PerformanceMonitor,
    ProductionInferenceEngine,
    ProductionMLService,
    ProductionModelLoader,
)


def make_history(days=100):

    dates = pd.date_range(
        "2025-01-01",
        periods=days,
        freq="D",
    )

    rows = []

    for outlet in [
        "O001",
        "O002",
    ]:

        for product in [
            "P001",
            "P002",
        ]:

            base = (
                20.0
                if product == "P001"
                else 30.0
            )

            for i, date in enumerate(
                dates
            ):

                demand = (
                    base
                    + 2.0
                    * np.sin(
                        i / 7.0
                    )
                    + (
                        3.0
                        if date.dayofweek
                        >= 5
                        else 0.0
                    )
                )

                rows.append(
                    {
                        "date": date,
                        "outlet_id": outlet,
                        "product_id": product,
                        "deconstrained_demand": demand,
                    }
                )

    return pd.DataFrame(rows)


def train_test_model(
    root: Path,
):

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = make_history()

    engine = ProductionInferenceEngine(
        ProductionModelLoader(
            model_root=str(root)
        )
    )

    features = engine.build_features(
        history
    )

    X = features[
        engine.FEATURE_COLUMNS
    ].dropna()

    y = features.loc[
        X.index,
        "deconstrained_demand",
    ]

    model = XGBRegressor(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    model.fit(
        X.astype(float),
        y.astype(float),
    )

    joblib.dump(
        model,
        root / "model.joblib",
    )

    (
        root / "manifest.json"
    ).write_text(
        json.dumps(
            {
                "model_name": "test_champion",
                "run_id": "test-version-001",
            }
        ),
        encoding="utf-8",
    )

    return history


def test_model_loader_caches(tmp_path):

    root = tmp_path / "champion"

    train_test_model(
        root
    )

    loader = ProductionModelLoader(
        model_root=str(root)
    )

    model1 = loader.load()
    load_time = loader.load_time_ms()

    model2 = loader.load()

    assert model1 is model2
    assert load_time >= 0
    assert loader.model_name() == (
        "test_champion"
    )
    assert loader.model_version() == (
        "test-version-001"
    )


def test_inference_feature_firewall(
    tmp_path,
):

    root = tmp_path / "champion"

    history = train_test_model(
        root
    )

    engine = ProductionInferenceEngine(
        ProductionModelLoader(
            model_root=str(root)
        )
    )

    features = engine.build_features(
        history
    )

    assert (
        "deconstrained_demand"
        not in engine.FEATURE_COLUMNS
    )

    assert (
        "future_demand"
        not in engine.FEATURE_COLUMNS
    )

    assert (
        "true_demand"
        not in engine.FEATURE_COLUMNS
    )

    assert "lag_1" in features.columns


def test_single_series_prediction(
    tmp_path,
):

    root = tmp_path / "champion"

    history = train_test_model(
        root
    )

    loader = ProductionModelLoader(
        model_root=str(root)
    )

    engine = ProductionInferenceEngine(
        loader
    )

    series = history[
        (history["outlet_id"] == "O001")
        & (history["product_id"] == "P001")
    ]

    prediction, elapsed = (
        engine.forecast_latest(
            series
        )
    )

    assert np.isfinite(
        prediction
    )

    assert prediction >= 0

    assert elapsed >= 0


def test_batch_inference(
    tmp_path,
):

    root = tmp_path / "champion"

    history = train_test_model(
        root
    )

    loader = ProductionModelLoader(
        model_root=str(root)
    )

    engine = ProductionInferenceEngine(
        loader
    )

    batch = BatchInferenceEngine(
        engine
    )

    result = batch.predict(
        history
    )

    assert len(result) == 4

    assert set(
        [
            "outlet_id",
            "product_id",
            "prediction",
            "inference_ms",
        ]
    ).issubset(
        result.columns
    )

    assert np.isfinite(
        result["prediction"]
    ).all()


def test_performance_monitor():

    monitor = PerformanceMonitor()

    report = monitor.benchmark(
        lambda: 1 + 1,
        iterations=5,
    )

    assert report.requests == 5
    assert (
        report.successful_requests
        == 5
    )
    assert (
        report.failed_requests
        == 0
    )
    assert report.mean_ms >= 0
    assert report.p95_ms >= 0
    assert report.p99_ms >= 0
    assert (
        report.requests_per_second
        > 0
    )


def test_production_service(
    tmp_path,
):

    root = tmp_path / "champion"

    history = train_test_model(
        root
    )

    service = ProductionMLService(
        model_root=str(root)
    )

    validation = service.validate()

    assert validation["passed"] is True

    response = service.forecast(
        "O001",
        "P001",
        history,
        horizon=1,
    )

    assert (
        response.outlet_id
        == "O001"
    )

    assert (
        response.product_id
        == "P001"
    )

    assert len(
        response.predictions
    ) == 1

    assert np.isfinite(
        response.predictions[0]
    )

    assert response.inference_ms >= 0


def test_service_rejects_invalid_horizon(
    tmp_path,
):

    root = tmp_path / "champion"

    history = train_test_model(
        root
    )

    service = ProductionMLService(
        model_root=str(root)
    )

    try:
        service.forecast(
            "O001",
            "P001",
            history,
            horizon=7,
        )

    except ValueError:
        assert True

    else:
        assert False


def test_missing_model_is_rejected(
    tmp_path,
):

    loader = ProductionModelLoader(
        model_root=str(
            tmp_path / "missing"
        )
    )

    assert loader.exists() is False

    try:
        loader.load()

    except FileNotFoundError:
        assert True

    else:
        assert False