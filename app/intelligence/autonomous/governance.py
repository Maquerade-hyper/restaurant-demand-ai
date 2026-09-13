from __future__ import annotations

from .schemas import (
    GovernanceDecision,
    HealthReport,
)


class AutonomousGovernance:
    """
    Commercial safety and governance gate.

    Autonomous ML may generate recommendations, but unhealthy
    models or low-confidence predictions cannot produce an
    autonomous commercial action.
    """

    def evaluate(
        self,
        *,
        health: HealthReport,
        action: str,
        confidence: float,
        high_risk: bool,
    ) -> GovernanceDecision:

        # -----------------------------------------------------
        # HEALTH GATE
        # -----------------------------------------------------

        if not (
            health.data_available
            and health.model_available
            and health.feature_contract_valid
            and health.prediction_valid
        ):

            return GovernanceDecision(
                action="HOLD",
                allowed=False,
                reason=(
                    "System health gate failed."
                ),
                risk_level="critical",
            )

        # -----------------------------------------------------
        # CONFIDENCE GATE
        # -----------------------------------------------------

        if confidence < 0.50:

            return GovernanceDecision(
                action="REVIEW_REQUIRED",
                allowed=False,
                reason=(
                    "Forecast confidence is below "
                    "the autonomous-action threshold."
                ),
                risk_level="high",
            )

        # -----------------------------------------------------
        # DRIFT GATE
        # -----------------------------------------------------

        if health.drift_signal:

            return GovernanceDecision(
                action="REVIEW_REQUIRED",
                allowed=False,
                reason=(
                    "Drift signal detected; autonomous "
                    "commercial action is held."
                ),
                risk_level="high",
            )

        # -----------------------------------------------------
        # HIGH-RISK GATE
        # -----------------------------------------------------

        if high_risk:

            return GovernanceDecision(
                action="REVIEW_REQUIRED",
                allowed=False,
                reason=(
                    "High-risk commercial decision "
                    "requires review."
                ),
                risk_level="high",
            )

        # -----------------------------------------------------
        # NO ACTION
        # -----------------------------------------------------

        if action == "NO_REPLENISHMENT":

            return GovernanceDecision(
                action=action,
                allowed=True,
                reason=(
                    "Inventory position satisfies "
                    "current target."
                ),
                risk_level="low",
            )

        # -----------------------------------------------------
        # NORMAL AUTONOMOUS ACTION
        # -----------------------------------------------------

        return GovernanceDecision(
            action=action,
            allowed=True,
            reason=(
                "Health, confidence and governance "
                "gates passed."
            ),
            risk_level="moderate",
        )