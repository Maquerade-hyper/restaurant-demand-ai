from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ClientDemoRequest:
    """
    Input contract for the client demonstration.

    The same contract can later be connected to a real client
    data adapter without changing the downstream intelligence output.
    """

    outlet_id: str
    product_id: str
    forecast_date: str
    horizon_days: int = 7

    scenario: str = "baseline"

    current_inventory: float | None = None
    lead_time_days: int = 3

    def __post_init__(self) -> None:
        if not self.outlet_id:
            raise ValueError("outlet_id is required")

        if not self.product_id:
            raise ValueError("product_id is required")

        if self.horizon_days not in {1, 3, 7}:
            raise ValueError("horizon_days must be one of 1, 3, 7")

        if self.lead_time_days < 0:
            raise ValueError("lead_time_days cannot be negative")

        if self.current_inventory is not None and self.current_inventory < 0:
            raise ValueError("current_inventory cannot be negative")


@dataclass
class ForecastOutput:
    d1: float
    d3: float
    d7: float
    daily_forecast: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DemandOutput:
    baseline_demand: float
    forecast_demand: float
    demand_change_pct: float
    spike_detected: bool
    spike_severity: str
    demand_risk: str
    confidence: float


@dataclass
class InventoryOutput:
    current_inventory: float
    required_inventory: float
    shortage: float
    safety_stock: float
    stockout_risk: str


@dataclass
class SupplyOutput:
    lead_time_days: int
    lead_time_demand: float
    recommended_order: float
    order_multiple: int
    supplier_risk: str


@dataclass
class AutonomousOutput:
    action: str
    priority: str
    reason: str


@dataclass
class ClientDemoResult:
    demo: Dict[str, Any]
    outlet: Dict[str, Any]
    product: Dict[str, Any]
    forecast: Dict[str, Any]
    demand_intelligence: Dict[str, Any]
    inventory: Dict[str, Any]
    supply: Dict[str, Any]
    autonomous_decision: Dict[str, Any]
    drivers: List[str]
    data_source: str
    production_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)