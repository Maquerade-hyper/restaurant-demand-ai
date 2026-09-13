from __future__ import annotations

import math

from .schemas import (
    AutonomousRequest,
)


class CommercialDecisionEngine:
    """
    Converts forecast + inventory + supplier constraints into
    a controlled commercial recommendation.

    This is a decision engine, not a claim that an order should
    automatically be placed with a supplier.
    """

    def decide(
        self,
        request: AutonomousRequest,
    ) -> dict:

        self._validate(request)

        adjusted_forecast = (
            request.forecast_demand
            * request.scenario_multiplier
        )

        adjusted_forecast = max(
            0.0,
            adjusted_forecast,
        )

        lead_time_demand = (
            adjusted_forecast
            * max(
                1.0,
                request.lead_time_days,
            )
        )

        inventory_position = (
            request.current_inventory
        )

        target_inventory = (
            lead_time_demand
            + request.safety_stock
        )

        shortage = max(
            0.0,
            target_inventory
            - inventory_position,
        )

        recommended_order = (
            shortage
        )

        # -----------------------------------------------------
        # MOQ
        # -----------------------------------------------------

        if (
            recommended_order > 0
            and request.minimum_order_quantity > 0
        ):
            recommended_order = max(
                recommended_order,
                request.minimum_order_quantity,
            )

        # -----------------------------------------------------
        # ORDER MULTIPLE
        # -----------------------------------------------------

        multiple = max(
            1.0,
            request.order_multiple,
        )

        if recommended_order > 0:

            recommended_order = (
                math.ceil(
                    recommended_order
                    / multiple
                )
                * multiple
            )

        # -----------------------------------------------------
        # RISK
        # -----------------------------------------------------

        if inventory_position <= 0:
            risk = "critical"

        elif inventory_position < lead_time_demand:
            risk = "high"

        elif (
            inventory_position
            < target_inventory
        ):
            risk = "moderate"

        else:
            risk = "low"

        # -----------------------------------------------------
        # ACTION
        # -----------------------------------------------------

        if risk == "critical":
            action = "URGENT_REPLENISHMENT"

        elif risk == "high":
            action = "REPLENISH"

        elif risk == "moderate":
            action = "MONITOR_AND_REPLENISH"

        else:
            action = "NO_REPLENISHMENT"

        return {
            "adjusted_forecast": adjusted_forecast,
            "lead_time_demand": lead_time_demand,
            "inventory_position": inventory_position,
            "target_inventory": target_inventory,
            "shortage": shortage,
            "recommended_order": recommended_order,
            "risk": risk,
            "action": action,
        }

    @staticmethod
    def _validate(
        request: AutonomousRequest,
    ):

        numeric_fields = {
            "forecast_demand":
                request.forecast_demand,
            "current_inventory":
                request.current_inventory,
            "lead_time_days":
                request.lead_time_days,
            "safety_stock":
                request.safety_stock,
            "minimum_order_quantity":
                request.minimum_order_quantity,
            "order_multiple":
                request.order_multiple,
            "scenario_multiplier":
                request.scenario_multiplier,
            "demand_confidence":
                request.demand_confidence,
        }

        for name, value in numeric_fields.items():

            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

        if request.forecast_demand < 0:
            raise ValueError(
                "forecast_demand cannot be negative."
            )

        if request.current_inventory < 0:
            raise ValueError(
                "current_inventory cannot be negative."
            )

        if request.lead_time_days <= 0:
            raise ValueError(
                "lead_time_days must be > 0."
            )

        if request.safety_stock < 0:
            raise ValueError(
                "safety_stock cannot be negative."
            )

        if request.scenario_multiplier <= 0:
            raise ValueError(
                "scenario_multiplier must be > 0."
            )

        if not (
            0.0
            <= request.demand_confidence
            <= 1.0
        ):
            raise ValueError(
                "demand_confidence must be in [0,1]."
            )