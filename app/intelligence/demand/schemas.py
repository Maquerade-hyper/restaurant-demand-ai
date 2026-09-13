from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemandObservation:
    outlet_id: str
    product_id: str
    date: object
    true_demand: float
    observed_sales: float
    lost_demand: float
    stockout: bool

    @property
    def fulfillment_rate(self) -> float:
        if self.true_demand <= 0:
            return 1.0

        return min(
            self.observed_sales / self.true_demand,
            1.0,
        )