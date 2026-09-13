from __future__ import annotations

from app.inventory.lead_time import (
    calculate_lead_time_demand,
)

from app.inventory.safety_stock import (
    calculate_safety_stock,
)

from app.inventory.reorder_point import (
    calculate_reorder_point,
    should_reorder,
)

from app.inventory.order_quantity import (
    calculate_order_quantity,
    round_to_pack_size,
)


class InventoryService:

    def recommend(
        self,
        product_id: str,
        outlet_id: str,
        daily_forecast,
        demand_std: float,
        lead_time_days: int,
        inventory_position: float,
        service_level: float = 0.95,
        review_period_days: int = 1,
        pack_size: float = 1.0,
    ) -> dict:

        lead_time_demand = (
            calculate_lead_time_demand(
                daily_forecast,
                lead_time_days,
            )
        )

        safety_stock = (
            calculate_safety_stock(
                demand_std,
                lead_time_days,
                service_level,
            )
        )

        reorder_point = (
            calculate_reorder_point(
                lead_time_demand,
                safety_stock,
            )
        )

        reorder = should_reorder(
            inventory_position,
            reorder_point,
        )

        forecast_demand = float(
            daily_forecast[0]
        )

        recommended_order = (
            calculate_order_quantity(
                reorder_point,
                inventory_position,
                forecast_demand,
                review_period_days,
            )
        )

        recommended_order = (
            round_to_pack_size(
                recommended_order,
                pack_size,
            )
        )

        return {
            "product_id": product_id,
            "outlet_id": outlet_id,
            "forecast_demand": forecast_demand,
            "lead_time_days": lead_time_days,
            "lead_time_demand": lead_time_demand,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "inventory_position": inventory_position,
            "reorder": reorder,
            "recommended_order_quantity": (
                recommended_order
            ),
            "stockout_risk": (
                inventory_position
                < lead_time_demand
            ),
        }