from __future__ import annotations

from .decisions import CommercialDecisionEngine
from .governance import AutonomousGovernance
from .health import AutonomousHealthMonitor
from .schemas import (
    AutonomousDecision,
    AutonomousRequest,
)


class AutonomousController:

    def __init__(self):

        self.health = (
            AutonomousHealthMonitor()
        )

        self.decisions = (
            CommercialDecisionEngine()
        )

        self.governance = (
            AutonomousGovernance()
        )

    def execute(
        self,
        request: AutonomousRequest,
        *,
        data_available: bool = True,
        model_available: bool = True,
        feature_contract_valid: bool = True,
        drift_signal: bool = False,
    ) -> AutonomousDecision:

        commercial = (
            self.decisions.decide(
                request
            )
        )

        health = self.health.evaluate(
            data_available=data_available,
            model_available=model_available,
            feature_contract_valid=(
                feature_contract_valid
            ),
            prediction=(
                commercial[
                    "adjusted_forecast"
                ]
            ),
            drift_signal=drift_signal,
        )

        governance = (
            self.governance.evaluate(
                health=health,
                action=commercial["action"],
                confidence=(
                    request.demand_confidence
                ),
                high_risk=request.high_risk,
            )
        )

        final_action = (
            governance.action
            if not governance.allowed
            else commercial["action"]
        )

        explanation = (
            f"Forecast={commercial['adjusted_forecast']:.2f}; "
            f"lead-time demand={commercial['lead_time_demand']:.2f}; "
            f"inventory={commercial['inventory_position']:.2f}; "
            f"shortage={commercial['shortage']:.2f}; "
            f"recommended order={commercial['recommended_order']:.2f}; "
            f"risk={commercial['risk']}; "
            f"governance={governance.reason}"
        )

        return AutonomousDecision(
            outlet_id=request.outlet_id,
            product_id=request.product_id,
            raw_forecast=(
                request.forecast_demand
            ),
            scenario_adjusted_forecast=(
                commercial[
                    "adjusted_forecast"
                ]
            ),
            current_inventory=(
                request.current_inventory
            ),
            lead_time_demand=(
                commercial[
                    "lead_time_demand"
                ]
            ),
            safety_stock=(
                request.safety_stock
            ),
            inventory_position=(
                commercial[
                    "inventory_position"
                ]
            ),
            shortage=(
                commercial[
                    "shortage"
                ]
            ),
            recommended_order=(
                commercial[
                    "recommended_order"
                ]
            ),
            stockout_risk=(
                commercial["risk"]
            ),
            commercial_action=(
                final_action
            ),
            confidence=(
                request.demand_confidence
            ),
            health=health,
            governance=governance,
            explanation=explanation,
        )