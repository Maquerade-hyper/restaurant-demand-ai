from __future__ import annotations

from typing import Iterable, List

from .schemas import DriftReport, RetrainingDecision


class RetrainingTrigger:
    """
    Converts monitoring signals into a retraining decision.

    Retraining can be triggered by:
      1. significant feature drift
      2. model error degradation
      3. sufficient scheduled new data

    The trigger does not promote a model.
    """

    def __init__(
        self,
        min_new_rows: int = 500,
        min_drifted_features: int = 1,
        error_degradation_threshold: float = 0.10,
    ):
        self.min_new_rows = int(min_new_rows)
        self.min_drifted_features = int(min_drifted_features)
        self.error_degradation_threshold = float(
            error_degradation_threshold
        )

    def evaluate(
        self,
        drift_reports: Iterable[DriftReport],
        new_rows: int,
        error_degradation: float = 0.0,
        scheduled: bool = False,
    ) -> RetrainingDecision:

        reports: List[DriftReport] = list(drift_reports)

        drifted = [
            report.feature
            for report in reports
            if report.drifted
        ]

        reasons = []

        if len(drifted) >= self.min_drifted_features:
            reasons.append(
                f"significant_drift:{len(drifted)}"
            )

        if error_degradation >= self.error_degradation_threshold:
            reasons.append(
                f"error_degradation:{error_degradation:.4f}"
            )

        if new_rows >= self.min_new_rows:
            reasons.append(
                f"new_data:{new_rows}"
            )

        if scheduled:
            reasons.append("scheduled_retraining")

        return RetrainingDecision(
            should_retrain=bool(reasons),
            reasons=reasons,
            drifted_features=drifted,
            error_degradation=float(error_degradation),
        )