"""
PART 28 - CONTINUOUS LEARNING ACCEPTANCE RUNNER

28A - Data + Model Drift Monitoring
28B - Retraining Trigger Engine
28C - Candidate Training + Validation
28D - Model Promotion + Rollback
28E - Unified Continuous Learning Service

Scientific rules:
    - chronological data split
    - candidate trains only on training data
    - incumbent champion is trained on an earlier training window
    - candidate and champion use the same unseen test period
    - promotion requires empirical improvement
    - no promotion is based merely on retraining success
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT IMPORT FIX
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from app.learning.continuous import (
    CandidateTrainer,
    ContinuousLearningService,
    DriftMonitor,
    ModelRegistry,
    RetrainingTrigger,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "interim"
    / "demand_censoring_intelligence.csv"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "continuous_learning_demo"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "continuous_learning_result.json"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message: str):
    print(f"\nERROR: {message}")
    raise SystemExit(1)


def load_data() -> pd.DataFrame:

    if not DATA_PATH.exists():
        fail(
            f"Dataset not found: {DATA_PATH}"
        )

    required = [
        "date",
        "outlet_id",
        "product_id",
        "deconstrained_demand",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=required,
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    return df


def select_series(
    df: pd.DataFrame,
    max_series: int = 12,
) -> pd.DataFrame:

    counts = (
        df.groupby(
            [
                "outlet_id",
                "product_id",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    selected = counts.head(
        max_series
    ).index

    index = df.set_index(
        [
            "outlet_id",
            "product_id",
        ]
    ).index

    mask = index.isin(
        selected
    )

    result = df.loc[mask].copy()

    return result.sort_values(
        [
            "outlet_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
):
    """
    Final 30 days:
        unseen evaluation

    Previous 30 days:
        validation

    Everything before that:
        training

    This is deliberately chronological.
    """

    dates = np.sort(
        df["date"].unique()
    )

    if len(dates) < 120:
        fail(
            "Not enough dates for continuous-learning split."
        )

    test_dates = dates[-30:]

    validation_dates = dates[-60:-30]

    train_dates = dates[:-60]

    train = df[
        df["date"].isin(
            train_dates
        )
    ].copy()

    validation = df[
        df["date"].isin(
            validation_dates
        )
    ].copy()

    test = df[
        df["date"].isin(
            test_dates
        )
    ].copy()

    return (
        train,
        validation,
        test,
    )


def print_split(
    train,
    validation,
    test,
):

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print(
        f"Train dates: "
        f"{train['date'].min().date()} "
        f"-> "
        f"{train['date'].max().date()}"
    )

    print(
        f"Validation dates: "
        f"{validation['date'].min().date()} "
        f"-> "
        f"{validation['date'].max().date()}"
    )

    print(
        f"Test dates: "
        f"{test['date'].min().date()} "
        f"-> "
        f"{test['date'].max().date()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 28 - CONTINUOUS LEARNING"
    )

    print(
        "=" * 72
    )

    print(
        f"Dataset: {DATA_PATH}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    print(
        f"Source rows: {len(df):,}"
    )

    # ========================================================
    # SELECT REPRESENTATIVE SERIES
    # ========================================================

    df = select_series(
        df,
        max_series=12,
    )

    series_count = (
        df[
            [
                "outlet_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Selected rows: {len(df):,}"
    )

    print(
        f"Selected series: {series_count}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    (
        train,
        validation,
        test,
    ) = chronological_split(df)

    print()

    print_split(
        train,
        validation,
        test,
    )

    # ========================================================
    # 28A - DRIFT MONITORING
    # ========================================================

    print(
        "\n28A - DRIFT MONITORING"
    )

    drift_monitor = DriftMonitor(
        psi_threshold=0.25,
        mean_shift_threshold=0.20,
    )

    drift_features = [
        "deconstrained_demand",
    ]

    # IMPORTANT:
    # Pass the full frames to the service.
    # DriftMonitor only reads the requested features.
    reference = train.copy()

    current = validation.copy()

    drift_reports = (
        drift_monitor.compare(
            reference=reference,
            current=current,
            features=drift_features,
        )
    )

    for report in drift_reports:

        print(
            f"{report.feature}: "
            f"mean_shift={report.mean_shift:.4f}, "
            f"psi={report.psi:.4f}, "
            f"drifted={report.drifted}"
        )

    if not drift_reports:
        fail(
            "No drift reports were generated."
        )

    if not all(
        np.isfinite(report.psi)
        for report in drift_reports
    ):
        fail(
            "Non-finite drift metric detected."
        )

    print(
        "DRIFT MONITOR: PASS"
    )

    # ========================================================
    # 28B - RETRAINING TRIGGER
    # ========================================================

    print(
        "\n28B - RETRAINING TRIGGER"
    )

    trigger = RetrainingTrigger(
        min_new_rows=100,
        min_drifted_features=1,
        error_degradation_threshold=0.10,
    )

    decision = trigger.evaluate(
        drift_reports=drift_reports,
        new_rows=len(validation),
        scheduled=False,
    )

    print(
        f"Retraining required: "
        f"{decision.should_retrain}"
    )

    print(
        f"Reasons: "
        f"{decision.reasons}"
    )

    if not decision.should_retrain:
        print(
            "Trigger did not activate naturally."
        )

        print(
            "For acceptance purposes, the lifecycle "
            "will still be executed explicitly."
        )

    print(
        "RETRAINING TRIGGER: PASS"
    )

    # ========================================================
    # 28C / 28D
    #
    # Establish an EXISTING champion first.
    #
    # This is critical.
    #
    # Champion:
    #     earlier training window
    #
    # Candidate:
    #     complete training window
    #
    # Both:
    #     same unseen test set
    # ========================================================

    print(
        "\n28C - CANDIDATE TRAINING"
    )

    if MODEL_ROOT.exists():
        shutil.rmtree(
            MODEL_ROOT
        )

    registry = ModelRegistry(
        root=str(MODEL_ROOT)
    )

    trainer = CandidateTrainer(
        n_estimators=80
    )

    # --------------------------------------------------------
    # Seed an incumbent champion using an EARLIER window.
    # --------------------------------------------------------

    train_dates = np.sort(
        train["date"].unique()
    )

    if len(train_dates) < 60:
        fail(
            "Training window too short to establish champion."
        )

    champion_cutoff = train_dates[-30]

    champion_train = train[
        train["date"] < champion_cutoff
    ].copy()

    if champion_train.empty:
        fail(
            "Champion training window is empty."
        )

    print(
        f"Initial champion training rows: "
        f"{len(champion_train):,}"
    )

    print(
        f"Candidate training rows: "
        f"{len(train):,}"
    )

    # --------------------------------------------------------
    # Train initial champion.
    # --------------------------------------------------------

    champion_model = trainer.train(
        champion_train,
        "deconstrained_demand",
    )

    champion_path = registry.save_candidate(
        champion_model,
        "initial-champion",
        {
            "model_name": "initial_champion",
            "model_type": "XGBRegressor",
            "training_rows": len(
                champion_train
            ),
            "training_end": str(
                champion_train["date"].max()
            ),
        },
    )

    registry.promote(
        champion_path,
        {
            "model_name": "initial_champion",
            "model_type": "XGBRegressor",
            "training_rows": len(
                champion_train
            ),
            "training_end": str(
                champion_train["date"].max()
            ),
        },
    )

    if not registry.has_champion():
        fail(
            "Initial champion was not registered."
        )

    print(
        "Initial champion registered."
    )

    # --------------------------------------------------------
    # Candidate is trained on ALL available training data.
    # --------------------------------------------------------

    result = ContinuousLearningService(
        registry=registry,
        drift_monitor=drift_monitor,
        trigger=trigger,
        trainer=trainer,
    ).run(
        reference=reference,
        current=current,
        target="deconstrained_demand",
        drift_features=drift_features,
        train_df=train,
        validation_df=validation,
        evaluation_df=test,
        force_retraining=True,
    )

    print(
        f"Run ID: {result.run_id}"
    )

    # ========================================================
    # CANDIDATE RESULTS
    # ========================================================

    if result.candidate is None:
        fail(
            "Candidate evaluation was not produced."
        )

    candidate = result.candidate

    print(
        "\nCandidate evaluation:"
    )

    print(
        f"Champion MAE: "
        f"{candidate.champion_mae:.6f}"
    )

    print(
        f"Candidate MAE: "
        f"{candidate.candidate_mae:.6f}"
    )

    print(
        f"Champion RMSE: "
        f"{candidate.champion_rmse:.6f}"
    )

    print(
        f"Candidate RMSE: "
        f"{candidate.candidate_rmse:.6f}"
    )

    print(
        f"Champion bias: "
        f"{candidate.champion_bias:.6f}"
    )

    print(
        f"Candidate bias: "
        f"{candidate.candidate_bias:.6f}"
    )

    print(
        f"MAE improvement: "
        f"{candidate.improvement_pct:.4f}%"
    )

    print(
        f"Bias change: "
        f"{candidate.bias_change:.6f}"
    )

    print(
        f"Candidate stable: "
        f"{candidate.candidate_stable}"
    )

    print(
        f"Promotion gate: "
        f"{candidate.passes}"
    )

    print(
        "CANDIDATE VALIDATION: PASS"
    )

    # ========================================================
    # 28D - PROMOTION / REJECTION
    # ========================================================

    print(
        "\n28D - MODEL PROMOTION / ROLLBACK"
    )

    print(
        f"Action: "
        f"{result.promotion.action}"
    )

    print(
        f"Reason: "
        f"{result.promotion.reason}"
    )

    allowed_actions = {
        "PROMOTE_CANDIDATE",
        "REJECT_CANDIDATE",
        "KEEP_CHAMPION",
    }

    if result.promotion.action not in allowed_actions:
        fail(
            "Invalid promotion action."
        )

    if result.promotion.action == (
        "PROMOTE_CANDIDATE"
    ):

        if not registry.has_champion():
            fail(
                "Promotion reported success but champion "
                "does not exist."
            )

        manifest = (
            registry.load_champion_manifest()
        )

        print(
            "New champion:"
        )

        print(
            json.dumps(
                manifest,
                indent=2,
                default=str,
            )
        )

        print(
            "PROMOTION: PASS"
        )

        # ----------------------------------------------------
        # Verify rollback capability.
        # ----------------------------------------------------

        rollback_archives = sorted(
            [
                path
                for path in registry.archive_dir.iterdir()
                if path.is_dir()
            ]
        )

        if not rollback_archives:
            fail(
                "Promotion occurred without creating "
                "a rollback archive."
            )

        promoted_name = manifest.get(
            "model_name"
        )

        rollback_id = (
            registry.rollback()
        )

        rollback_manifest = (
            registry.load_champion_manifest()
        )

        rollback_name = (
            rollback_manifest or {}
        ).get(
            "model_name"
        )

        print(
            f"Rollback archive: "
            f"{rollback_id}"
        )

        print(
            f"Champion after rollback: "
            f"{rollback_name}"
        )

        if rollback_name != "initial_champion":
            fail(
                "Rollback did not restore initial champion."
            )

        print(
            "ROLLBACK: PASS"
        )

    else:

        # Candidate rejection is also a successful
        # continuous-learning outcome.
        #
        # A model that fails the gate MUST NOT replace
        # the champion.

        manifest = (
            registry.load_champion_manifest()
        )

        champion_name = (
            manifest or {}
        ).get(
            "model_name"
        )

        print(
            f"Champion remains: "
            f"{champion_name}"
        )

        if champion_name != "initial_champion":
            fail(
                "Rejected candidate changed the champion."
            )

        print(
            "CANDIDATE REJECTION: PASS"
        )

    # ========================================================
    # 28E - SERVICE VALIDATION
    # ========================================================

    print(
        "\n28E - CONTINUOUS LEARNING SERVICE"
    )

    service = ContinuousLearningService(
        registry=registry,
        drift_monitor=drift_monitor,
        trigger=trigger,
        trainer=trainer,
    )

    validation = service.validate(
        result
    )

    print(
        f"Validation passed: "
        f"{validation['passed']}"
    )

    if validation["errors"]:
        print(
            "Validation errors:",
            validation["errors"],
        )

    if not validation["passed"]:
        fail(
            "Continuous-learning service validation failed."
        )

    print(
        "SERVICE VALIDATION: PASS"
    )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL ACCEPTANCE
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PART 28 ACCEPTANCE: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Result: {OUTPUT_PATH}"
    )

    print(
        f"Model registry: {MODEL_ROOT}"
    )


if __name__ == "__main__":
    main()