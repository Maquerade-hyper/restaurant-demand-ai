import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.learning.continuous.candidate import CandidateTrainer
from app.learning.continuous.drift import DriftMonitor
from app.learning.continuous.registry import ModelRegistry
from app.learning.continuous.service import (
    ContinuousLearningService,
)
from app.learning.continuous.triggers import (
    RetrainingTrigger,
)


def make_data(days=100, start="2025-01-01"):
    dates = pd.date_range(
        start,
        periods=days,
        freq="D",
    )

    rows = []

    for outlet in ["O001", "O002"]:
        for product in ["P001", "P002"]:
            base = 20.0 if product == "P001" else 30.0

            for i, date in enumerate(dates):
                value = (
                    base
                    + 2.0 * np.sin(i / 7.0)
                    + (3.0 if date.dayofweek >= 5 else 0.0)
                )

                rows.append(
                    {
                        "date": date,
                        "outlet_id": outlet,
                        "product_id": product,
                        "deconstrained_demand": value,
                    }
                )

    return pd.DataFrame(rows)


def test_drift_monitor_no_drift():
    monitor = DriftMonitor()

    reference = pd.DataFrame(
        {"x": np.ones(100)}
    )

    current = pd.DataFrame(
        {"x": np.ones(100)}
    )

    reports = monitor.compare(
        reference,
        current,
        ["x"],
    )

    assert len(reports) == 1
    assert reports[0].psi == 0.0
    assert reports[0].drifted is False


def test_drift_monitor_detects_shift():
    monitor = DriftMonitor(
        mean_shift_threshold=0.20,
    )

    reference = pd.DataFrame(
        {"x": np.ones(100) * 10}
    )

    current = pd.DataFrame(
        {"x": np.ones(100) * 20}
    )

    reports = monitor.compare(
        reference,
        current,
        ["x"],
    )

    assert reports[0].drifted is True
    assert reports[0].mean_shift >= 0.20


def test_error_degradation():
    result = DriftMonitor.error_degradation(
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 10],
        [12, 12, 12],
    )

    assert result > 0


def test_trigger_no_retraining():
    trigger = RetrainingTrigger(
        min_new_rows=500,
    )

    decision = trigger.evaluate(
        drift_reports=[],
        new_rows=10,
    )

    assert decision.should_retrain is False


def test_trigger_new_data():
    trigger = RetrainingTrigger(
        min_new_rows=100,
    )

    decision = trigger.evaluate(
        drift_reports=[],
        new_rows=100,
    )

    assert decision.should_retrain is True
    assert "new_data:100" in decision.reasons


def test_trigger_drift():
    trigger = RetrainingTrigger()

    class Report:
        feature = "x"
        drifted = True

    decision = trigger.evaluate(
        drift_reports=[Report()],
        new_rows=0,
    )

    assert decision.should_retrain is True
    assert "x" in decision.drifted_features


def test_candidate_feature_firewall():
    trainer = CandidateTrainer()

    data = make_data()

    features = trainer.build_features(
        data,
        "deconstrained_demand",
    )

    for forbidden in [
        "future_demand",
        "true_demand",
        "lost_demand",
        "prediction",
    ]:
        assert forbidden not in trainer.FEATURE_COLUMNS

    assert "lag_1" in features.columns
    assert "rolling_mean_7" in features.columns


def test_candidate_training_and_prediction():
    trainer = CandidateTrainer(
        n_estimators=10,
    )

    data = make_data(100)

    model = trainer.train(
        data.iloc[:320],
        "deconstrained_demand",
    )

    prediction = trainer.predict(
        model,
        data.iloc[320:],
        "deconstrained_demand",
    )

    assert len(prediction) == len(data.iloc[320:])
    assert np.isfinite(
        prediction[
            np.isfinite(prediction)
        ]
    ).all()


def test_candidate_evaluation_gate():
    trainer = CandidateTrainer()

    actual = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    champion = np.array(
        [12.0, 22.0, 32.0, 42.0]
    )

    candidate = np.array(
        [10.5, 20.5, 30.5, 40.5]
    )

    result = trainer.evaluate_candidate(
        champion,
        candidate,
        actual,
    )

    assert result.improvement_pct > 0
    assert result.candidate_stable is True
    assert result.passes is True


def test_candidate_rejected_when_worse():
    trainer = CandidateTrainer()

    actual = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    champion = np.array(
        [10.1, 20.1, 30.1, 40.1]
    )

    candidate = np.array(
        [20.0, 30.0, 40.0, 50.0]
    )

    result = trainer.evaluate_candidate(
        champion,
        candidate,
        actual,
    )

    assert result.passes is False
    assert len(result.reasons) > 0


def test_registry_candidate_and_promotion(tmp_path):
    registry = ModelRegistry(
        root=str(tmp_path / "models")
    )

    trainer = CandidateTrainer(
        n_estimators=5
    )

    data = make_data(100)

    model = trainer.train(
        data.iloc[:320],
        "deconstrained_demand",
    )

    candidate_path = registry.save_candidate(
        model,
        "test-run",
        {
            "model_name": "candidate",
            "model_type": "xgboost",
        },
    )

    assert candidate_path.exists()
    assert registry.has_champion() is False

    registry.promote(
        candidate_path,
        {
            "model_name": "candidate",
            "model_type": "xgboost",
        },
    )

    assert registry.has_champion() is True
    assert registry.load_champion() is not None

    manifest = registry.load_champion_manifest()

    assert manifest["model_name"] == "candidate"


def test_registry_rollback(tmp_path):
    registry = ModelRegistry(
        root=str(tmp_path / "models")
    )

    trainer = CandidateTrainer(
        n_estimators=5
    )

    data = make_data(100)

    model1 = trainer.train(
        data.iloc[:320],
        "deconstrained_demand",
    )

    candidate1 = registry.save_candidate(
        model1,
        "run-1",
        {"model_name": "model-1"},
    )

    registry.promote(
        candidate1,
        {"model_name": "model-1"},
    )

    model2 = trainer.train(
        data.iloc[:340],
        "deconstrained_demand",
    )

    candidate2 = registry.save_candidate(
        model2,
        "run-2",
        {"model_name": "model-2"},
    )

    registry.promote(
        candidate2,
        {"model_name": "model-2"},
    )

    assert (
        registry.load_champion_manifest()
        ["model_name"]
        == "model-2"
    )

    registry.rollback()

    assert (
        registry.load_champion_manifest()
        ["model_name"]
        == "model-1"
    )


def test_service_validation():
    service = ContinuousLearningService()

    data = make_data(20)

    reports = service.monitor(
        data,
        data.copy(),
        ["deconstrained_demand"],
    )

    assert len(reports) == 1

    decision = service.decide_retraining(
        reports,
        new_rows=0,
    )

    assert decision.should_retrain is False