from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .candidate import CandidateTrainer
from .drift import DriftMonitor
from .registry import ModelRegistry
from .schemas import (
    CandidateEvaluation,
    LearningRunResult,
    PromotionDecision,
)
from .triggers import RetrainingTrigger


class ContinuousLearningService:
    """
    End-to-end continuous-learning controller.

    Flow:

        reference/current
              ↓
        drift monitoring
              ↓
        retraining trigger
              ↓
        candidate training
              ↓
        common unseen holdout
              ↓
        promotion gate
              ↓
        registry
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        trigger: Optional[RetrainingTrigger] = None,
        trainer: Optional[CandidateTrainer] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.drift_monitor = (
            drift_monitor or DriftMonitor()
        )
        self.trigger = (
            trigger or RetrainingTrigger()
        )
        self.trainer = (
            trainer or CandidateTrainer()
        )

    @staticmethod
    def _validate_dataset(
        df: pd.DataFrame,
        target: str,
    ) -> None:

        required = {
            "date",
            "outlet_id",
            "product_id",
            target,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

    def monitor(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: Iterable[str],
    ):
        return self.drift_monitor.compare(
            reference,
            current,
            features,
        )

    def decide_retraining(
        self,
        drift_reports,
        new_rows: int,
        error_degradation: float = 0.0,
        scheduled: bool = False,
    ):
        return self.trigger.evaluate(
            drift_reports=drift_reports,
            new_rows=new_rows,
            error_degradation=error_degradation,
            scheduled=scheduled,
        )

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target: str,
        drift_features: Iterable[str],
        train_df: Optional[pd.DataFrame] = None,
        validation_df: Optional[pd.DataFrame] = None,
        evaluation_df: Optional[pd.DataFrame] = None,
        force_retraining: bool = False,
    ) -> LearningRunResult:

        self._validate_dataset(
            reference,
            target,
        )

        self._validate_dataset(
            current,
            target,
        )

        run_id = uuid.uuid4().hex[:12]

        drift_reports = self.monitor(
            reference,
            current,
            drift_features,
        )

        retraining = self.decide_retraining(
            drift_reports=drift_reports,
            new_rows=len(current),
            scheduled=force_retraining,
        )

        if not retraining.should_retrain:
            return LearningRunResult(
                drift=drift_reports,
                retraining=retraining,
                candidate=None,
                promotion=PromotionDecision(
                    action="KEEP_CHAMPION",
                    model_name=None,
                    reason="Retraining trigger not activated.",
                ),
                run_id=run_id,
            )

        if train_df is None:
            raise ValueError(
                "train_df required when retraining is triggered."
            )

        if validation_df is None:
            raise ValueError(
                "validation_df required when retraining is triggered."
            )

        if evaluation_df is None:
            raise ValueError(
                "evaluation_df required when retraining is triggered."
            )

        # Candidate is trained only on the training period.
        candidate_model = self.trainer.train(
            train_df,
            target,
        )

        candidate_validation_pred = (
            self.trainer.predict(
                candidate_model,
                validation_df,
                target,
            )
        )

        # Candidate must have valid predictions before proceeding.
        validation_mask = (
            np.isfinite(candidate_validation_pred)
            & np.isfinite(
                validation_df[target].to_numpy(
                    dtype=float
                )
            )
        )

        if validation_mask.sum() == 0:
            raise ValueError(
                "Candidate produced no valid validation predictions."
            )

        # Train incumbent/champion if registry does not yet contain one.
        champion_model = (
            self.registry.load_champion()
        )

        if champion_model is None:
            champion_model = self.trainer.train(
                train_df,
                target,
            )

            champion_name = "initial_champion"

        else:
            manifest = (
                self.registry.load_champion_manifest()
                or {}
            )

            champion_name = manifest.get(
                "model_name",
                "champion",
            )

        # Common unseen evaluation period.
        candidate_pred = self.trainer.predict(
            candidate_model,
            evaluation_df,
            target,
        )

        champion_pred = self.trainer.predict(
            champion_model,
            evaluation_df,
            target,
        )

        actual = evaluation_df[target].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(actual)
            & np.isfinite(candidate_pred)
            & np.isfinite(champion_pred)
        )

        if valid.sum() == 0:
            raise ValueError(
                "No valid common evaluation rows."
            )

        evaluation = self.trainer.evaluate_candidate(
            champion_predictions=champion_pred[valid],
            candidate_predictions=candidate_pred[valid],
            actual=actual[valid],
            candidate_name="xgboost_candidate",
        )

        manifest = {
            "run_id": run_id,
            "model_name": evaluation.model_name,
            "model_type": "XGBRegressor",
            "candidate_mae": evaluation.candidate_mae,
            "candidate_rmse": evaluation.candidate_rmse,
            "candidate_bias": evaluation.candidate_bias,
            "champion_mae": evaluation.champion_mae,
            "champion_rmse": evaluation.champion_rmse,
            "champion_bias": evaluation.champion_bias,
            "improvement_pct": evaluation.improvement_pct,
            "feature_columns": self.trainer.FEATURE_COLUMNS,
        }

        candidate_path = self.registry.save_candidate(
            candidate_model,
            run_id,
            manifest,
        )

        if evaluation.passes:
            timestamp = self.registry.promote(
                candidate_path,
                manifest,
            )

            promotion = PromotionDecision(
                action="PROMOTE_CANDIDATE",
                model_name=evaluation.model_name,
                reason=(
                    "Candidate passed empirical promotion gates."
                ),
                previous_champion=champion_name,
            )

            artifact_path = str(
                candidate_path
            )

        else:
            promotion = PromotionDecision(
                action="REJECT_CANDIDATE",
                model_name=evaluation.model_name,
                reason=(
                    "Candidate failed empirical promotion gates: "
                    + "; ".join(
                        evaluation.reasons
                    )
                ),
                previous_champion=champion_name,
            )

            artifact_path = str(
                candidate_path
            )

        return LearningRunResult(
            drift=drift_reports,
            retraining=retraining,
            candidate=evaluation,
            promotion=promotion,
            run_id=run_id,
            artifact_path=artifact_path,
        )

    def validate(
        self,
        result: LearningRunResult,
    ) -> dict:

        errors = []

        if not result.run_id:
            errors.append(
                "missing_run_id"
            )

        for report in result.drift:
            if not np.isfinite(report.psi):
                errors.append(
                    f"nonfinite_psi:{report.feature}"
                )

            if report.psi < 0:
                errors.append(
                    f"negative_psi:{report.feature}"
                )

        if result.candidate is not None:
            candidate = result.candidate

            if not np.isfinite(
                candidate.candidate_mae
            ):
                errors.append(
                    "nonfinite_candidate_mae"
                )

            if not np.isfinite(
                candidate.champion_mae
            ):
                errors.append(
                    "nonfinite_champion_mae"
                )

            if not np.isfinite(
                candidate.improvement_pct
            ):
                errors.append(
                    "nonfinite_improvement"
                )

        allowed_actions = {
            "KEEP_CHAMPION",
            "PROMOTE_CANDIDATE",
            "REJECT_CANDIDATE",
        }

        if result.promotion.action not in allowed_actions:
            errors.append(
                "invalid_promotion_action"
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }