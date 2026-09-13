from __future__ import annotations

import math

from .schemas import HealthReport


class AutonomousHealthMonitor:
    """
    Lightweight production health gate.

    This layer does not make claims about model accuracy.
    It checks whether the system is healthy enough to produce
    an autonomous commercial decision.
    """

    def evaluate(
        self,
        *,
        data_available: bool,
        model_available: bool,
        feature_contract_valid: bool,
        prediction: float,
        drift_signal: bool = False,
    ) -> HealthReport:

        errors = []

        prediction_valid = (
            math.isfinite(prediction)
            and prediction >= 0
        )

        if not data_available:
            errors.append(
                "data_unavailable"
            )

        if not model_available:
            errors.append(
                "model_unavailable"
            )

        if not feature_contract_valid:
            errors.append(
                "feature_contract_invalid"
            )

        if not prediction_valid:
            errors.append(
                "prediction_invalid"
            )

        return HealthReport(
            data_available=data_available,
            model_available=model_available,
            feature_contract_valid=(
                feature_contract_valid
            ),
            prediction_valid=prediction_valid,
            drift_signal=drift_signal,
            errors=errors,
        )

    @staticmethod
    def healthy(
        report: HealthReport,
    ) -> bool:

        return (
            report.data_available
            and report.model_available
            and report.feature_contract_valid
            and report.prediction_valid
        )