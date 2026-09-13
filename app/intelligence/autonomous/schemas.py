from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AutonomousRequest:
    outlet_id: str
    product_id: str

    forecast_demand: float

    current_inventory: float = 0.0
    lead_time_days: float = 1.0
    safety_stock: float = 0.0

    minimum_order_quantity: float = 0.0
    order_multiple: float = 1.0

    scenario_multiplier: float = 1.0

    demand_confidence: float = 1.0

    high_risk: bool = False


@dataclass
class HealthReport:
    data_available: bool
    model_available: bool
    feature_contract_valid: bool
    prediction_valid: bool
    drift_signal: bool

    errors: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_available": self.data_available,
            "model_available": self.model_available,
            "feature_contract_valid": (
                self.feature_contract_valid
            ),
            "prediction_valid": (
                self.prediction_valid
            ),
            "drift_signal": self.drift_signal,
            "errors": list(self.errors),
        }


@dataclass
class GovernanceDecision:
    action: str
    allowed: bool
    reason: str
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_level": self.risk_level,
        }


@dataclass
class AutonomousDecision:
    outlet_id: str
    product_id: str

    raw_forecast: float
    scenario_adjusted_forecast: float

    current_inventory: float
    lead_time_demand: float
    safety_stock: float

    inventory_position: float
    shortage: float

    recommended_order: float

    stockout_risk: str
    commercial_action: str

    confidence: float

    health: HealthReport
    governance: GovernanceDecision

    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outlet_id": self.outlet_id,
            "product_id": self.product_id,
            "raw_forecast": self.raw_forecast,
            "scenario_adjusted_forecast": (
                self.scenario_adjusted_forecast
            ),
            "current_inventory": self.current_inventory,
            "lead_time_demand": self.lead_time_demand,
            "safety_stock": self.safety_stock,
            "inventory_position": (
                self.inventory_position
            ),
            "shortage": self.shortage,
            "recommended_order": (
                self.recommended_order
            ),
            "stockout_risk": self.stockout_risk,
            "commercial_action": (
                self.commercial_action
            ),
            "confidence": self.confidence,
            "health": self.health.to_dict(),
            "governance": (
                self.governance.to_dict()
            ),
            "explanation": self.explanation,
        }