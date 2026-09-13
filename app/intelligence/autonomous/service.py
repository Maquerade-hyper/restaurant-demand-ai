from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .controller import AutonomousController
from .schemas import (
    AutonomousDecision,
    AutonomousRequest,
)


class AutonomousCommercialMLService:
    """
    Final commercial ML orchestration service.

    The service intentionally separates:
        prediction
        commercial calculation
        governance
        final action

    This makes every autonomous decision auditable.
    """

    def __init__(self):

        self.controller = (
            AutonomousController()
        )

    def decide(
        self,
        request: AutonomousRequest,
        *,
        data_available: bool = True,
        model_available: bool = True,
        feature_contract_valid: bool = True,
        drift_signal: bool = False,
    ) -> AutonomousDecision:

        return self.controller.execute(
            request,
            data_available=data_available,
            model_available=model_available,
            feature_contract_valid=(
                feature_contract_valid
            ),
            drift_signal=drift_signal,
        )

    def decide_dict(
        self,
        request: AutonomousRequest,
        **kwargs,
    ) -> Dict:

        return self.decide(
            request,
            **kwargs,
        ).to_dict()

    def batch(
        self,
        requests: Iterable[
            AutonomousRequest
        ],
    ) -> List[Dict]:

        results = []

        for request in requests:

            results.append(
                self.decide_dict(
                    request
                )
            )

        return results

    def validate(
        self,
        decision: AutonomousDecision,
    ) -> dict:

        errors = []

        if not decision.outlet_id:
            errors.append(
                "missing_outlet_id"
            )

        if not decision.product_id:
            errors.append(
                "missing_product_id"
            )

        numeric_values = {
            "raw_forecast":
                decision.raw_forecast,
            "scenario_adjusted_forecast":
                decision.scenario_adjusted_forecast,
            "lead_time_demand":
                decision.lead_time_demand,
            "current_inventory":
                decision.current_inventory,
            "recommended_order":
                decision.recommended_order,
            "confidence":
                decision.confidence,
        }

        for name, value in numeric_values.items():

            if not math.isfinite(
                float(value)
            ):
                errors.append(
                    f"non_finite:{name}"
                )

        if decision.recommended_order < 0:
            errors.append(
                "negative_recommended_order"
            )

        if not (
            0.0
            <= decision.confidence
            <= 1.0
        ):
            errors.append(
                "invalid_confidence"
            )

        if decision.health.errors:
            errors.extend(
                decision.health.errors
            )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }