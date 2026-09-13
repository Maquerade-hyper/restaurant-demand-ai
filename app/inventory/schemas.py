from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InventoryPosition:
    opening_stock: float
    received_stock: float
    closing_stock: float
    incoming_stock: float
    reserved_stock: float = 0.0

    @property
    def available_stock(self) -> float:
        return max(
            self.closing_stock
            + self.incoming_stock
            - self.reserved_stock,
            0.0,
        )


@dataclass
class InventoryRecommendation:
    product_id: str
    outlet_id: str

    forecast_demand: float
    lead_time_days: int

    lead_time_demand: float
    safety_stock: float
    reorder_point: float

    inventory_position: float
    recommended_order_quantity: float

    stockout_risk: bool