from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.learning.runtime.data_gateway import (
    ClientDataGateway,
)
from app.learning.runtime.training_adapter import (
    ProductionTrainingAdapter,
)
from app.learning.runtime.runtime_service import (
    LearningRuntimeService,
)


# ============================================================
# TEST DATA
# ============================================================

def make_client_data(
    rows_per_series: int = 100,
    series_count: int = 12,
) -> pd.DataFrame:

    dates = pd.date_range(
        "2025-01-01",
        periods=rows_per_series,
        freq="D",
    )

    records = []

    for series_index in range(
        series_count
    ):

        outlet_id = (
            f"CLIENT_O{series_index + 1:03d}"
        )

        product_id = (
            f"CLIENT_P{series_index + 1:03d}"
        )

        for day_index, date in enumerate(
            dates
        ):

            weekly = (
                4.0
                if date.dayofweek >= 5
                else 0.0
            )

            trend = (
                day_index * 0.03
            )

            seasonal = np.sin(
                day_index / 7.0
            ) * 2.0

            quantity = max(
                1.0,
                30.0
                + series_index
                + weekly
                + trend
                + seasonal,
            )

            records.append(
                {
                    "date": date,
                    "outlet_id": outlet_id,
                    "product_id": product_id,
                    "quantity_sold": quantity,
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# GATEWAY TESTS
# ============================================================

def test_gateway_discovers_real_client_package(
    tmp_path,
):

    incoming = (
        tmp_path
        / "INCOMING"
    )

    gateway = ClientDataGateway(
        incoming_path=str(
            incoming
        ),
        processed_path=str(
            tmp_path / "PROCESSED"
        ),
        rejected_path=str(
            tmp_path / "REJECTED"
        ),
        archive_path=str(
            tmp_path / "ARCHIVE"
        ),
    )

    data = make_client_data(
        rows_per_series=10,
        series_count=2,
    )

    path = (
        incoming
        / "sales.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    packages = (
        gateway.discover_packages()
    )

    assert len(packages) == 1

    package = packages[0]

    assert package.source_type == (
        "real_client"
    )

    assert package.package_id
    assert path in package.files
    assert package.row_estimate == len(
        data
    )


def test_gateway_ignores_unsupported_files(
    tmp_path,
):

    incoming = (
        tmp_path
        / "INCOMING"
    )

    gateway = ClientDataGateway(
        incoming_path=str(
            incoming
        ),
        processed_path=str(
            tmp_path / "PROCESSED"
        ),
        rejected_path=str(
            tmp_path / "REJECTED"
        ),
        archive_path=str(
            tmp_path / "ARCHIVE"
        ),
    )

    (
        incoming
        / "notes.txt"
    ).write_text(
        "not training data",
        encoding="utf-8",
    )

    packages = (
        gateway.discover_packages()
    )

    assert packages == []


def test_gateway_rejects_invalid_schema(
    tmp_path,
):

    incoming = (
        tmp_path
        / "INCOMING"
    )

    gateway = ClientDataGateway(
        incoming_path=str(
            incoming
        ),
        processed_path=str(
            tmp_path / "PROCESSED"
        ),
        rejected_path=str(
            tmp_path / "REJECTED"
        ),
        archive_path=str(
            tmp_path / "ARCHIVE"
        ),
    )

    invalid = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=5,
            ),
            "sales": [1, 2, 3, 4, 5],
        }
    )

    path = (
        incoming
        / "invalid.csv"
    )

    invalid.to_csv(
        path,
        index=False,
    )

    packages = (
        gateway.discover_packages()
    )

    assert len(packages) == 1

    validation = (
        gateway.validate_package_sample(
            packages[0]
        )
    )

    assert validation["passed"] is False
    assert validation["errors"]

    rejected = (
        gateway.reject_package(
            packages[0],
            "INVALID_CLIENT_DATA",
        )
    )

    assert rejected.exists()
    assert not path.exists()
    assert (
        rejected / "manifest.json"
    ).exists()


# ============================================================
# RUNTIME DATA GATES
# ============================================================

def test_runtime_waits_for_insufficient_data(
    tmp_path,
):

    incoming = (
        tmp_path
        / "INCOMING"
    )

    gateway = ClientDataGateway(
        incoming_path=str(
            incoming
        ),
        processed_path=str(
            tmp_path / "PROCESSED"
        ),
        rejected_path=str(
            tmp_path / "REJECTED"
        ),
        archive_path=str(
            tmp_path / "ARCHIVE"
        ),
    )

    data = make_client_data(
        rows_per_series=30,
        series_count=2,
    )

    path = (
        incoming
        / "sales.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    service = LearningRuntimeService(
        gateway=gateway,
        minimum_rows=1000,
    )

    result = (
        service.scan_once()
    )

    assert result["packages"] == 1
    assert result["results"][0][
        "action"
    ] == "WAITING"

    assert "INSUFFICIENT_DATA" in (
        result["results"][0][
            "reason"
        ]
    )

    # Insufficient data must remain available.
    assert path.exists()


# ============================================================
# PRODUCTION FEATURE CONTRACT
# ============================================================

def test_production_adapter_uses_production_feature_pipeline():

    data = make_client_data(
        rows_per_series=150,
        series_count=12,
    )

    featured = (
        ProductionTrainingAdapter.build_features(
            data
        )
    )

    assert not featured.empty

    assert "quantity_sold" in (
        featured.columns
    )

    # Part 9 production contract must be
    # materially larger than the old 16-feature
    # continuous-learning experiment.
    excluded = {
        "quantity_sold",
        "date",
        "outlet_id",
        "product_id",
        "unit",
        "revenue",
        "actual_demand",
        "true_demand",
        "lost_demand",
        "future_demand",
        "closing_stock",
        "stockout",
        "demand_truth",
        "lost_demand_truth",
        "prediction",
        "target",
    }

    numeric_features = [
        column
        for column in featured.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            featured[column]
        )
    ]

    assert len(
        numeric_features
    ) >= 40


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def test_chronological_split_has_no_time_overlap():

    data = make_client_data(
        rows_per_series=120,
        series_count=12,
    )

    (
        train_df,
        validation_df,
        evaluation_df,
    ) = (
        ProductionTrainingAdapter.chronological_split(
            data,
            validation_days=30,
            evaluation_days=30,
        )
    )

    assert (
        train_df["date"].max()
        < validation_df["date"].min()
    )

    assert (
        validation_df["date"].max()
        < evaluation_df["date"].min()
    )

    assert (
        len(train_df) > 0
    )

    assert (
        len(validation_df) > 0
    )

    assert (
        len(evaluation_df) > 0
    )


# ============================================================
# MODEL TRAINING CONTRACT
# ============================================================

def test_production_model_trains_with_large_feature_contract():

    data = make_client_data(
        rows_per_series=150,
        series_count=12,
    )

    model = (
        ProductionTrainingAdapter.train_model(
            data
        )
    )

    assert model is not None

    assert len(
        model.feature_columns
    ) >= 40

    predictions = (
        ProductionTrainingAdapter.predict(
            model,
            data.tail(120),
        )
    )

    assert len(
        predictions
    ) == 120

    finite = np.isfinite(
        predictions
    )

    assert finite.any()

    assert (
        np.nanmin(predictions)
        >= 0
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

def test_initial_champion_is_saved_and_reloadable(
    tmp_path,
):

    data = make_client_data(
        rows_per_series=150,
        series_count=12,
    )

    registry = (
        __import__(
            "app.learning.continuous.registry",
            fromlist=[
                "ModelRegistry"
            ],
        ).ModelRegistry(
            root=str(
                tmp_path / "models"
            )
        )
    )

    model, run_id = (
        ProductionTrainingAdapter.create_initial_champion(
            data,
            registry,
        )
    )

    assert model is not None
    assert run_id

    assert registry.has_champion()

    loaded = (
        registry.load_champion()
    )

    assert loaded is not None

    assert len(
        loaded.feature_columns
    ) == len(
        model.feature_columns
    )

    manifest = (
        registry.load_champion_manifest()
    )

    assert manifest is not None

    assert manifest[
        "feature_count"
    ] >= 40


# ============================================================
# EVALUATION GATE
# ============================================================

def test_candidate_evaluation_rejects_non_improvement():

    actual = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    champion = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    candidate = np.array(
        [12, 22, 32, 42, 52],
        dtype=float,
    )

    evaluation = (
        ProductionTrainingAdapter.metrics(
            actual,
            candidate,
        )
    )

    champion_metrics = (
        ProductionTrainingAdapter.metrics(
            actual,
            champion,
        )
    )

    assert evaluation["mae"] > (
        champion_metrics["mae"]
    )

    improvement = (
        (
            champion_metrics["mae"]
            - evaluation["mae"]
        )
        / max(
            champion_metrics["mae"],
            1e-12,
        )
        * 100.0
    )

    assert improvement < 0


# ============================================================
# NO SYNTHETIC DATA PATH
# ============================================================

def test_gateway_isolated_from_synthetic_data(
    tmp_path,
):

    incoming = (
        tmp_path
        / "INCOMING"
    )

    gateway = ClientDataGateway(
        incoming_path=str(
            incoming
        ),
        processed_path=str(
            tmp_path / "PROCESSED"
        ),
        rejected_path=str(
            tmp_path / "REJECTED"
        ),
        archive_path=str(
            tmp_path / "ARCHIVE"
        ),
    )

    # Synthetic/interim paths are deliberately
    # not supplied to the gateway.
    assert "synthetic" not in str(
        gateway.incoming_path
    ).lower()

    assert "interim" not in str(
        gateway.incoming_path
    ).lower()

    assert (
        gateway.discover_packages()
        == []
    )