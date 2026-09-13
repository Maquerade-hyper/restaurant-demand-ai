from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.intelligence.autonomous import (
    AutonomousCommercialMLService,
    AutonomousRequest,
)


router = APIRouter(
    prefix="/api/v1/autonomous",
    tags=["autonomous-ml"],
)

service = (
    AutonomousCommercialMLService()
)


class AutonomousDecisionRequest(BaseModel):

    outlet_id: str
    product_id: str

    forecast_demand: float = Field(
        ge=0
    )

    current_inventory: float = Field(
        default=0,
        ge=0,
    )

    lead_time_days: float = Field(
        default=1,
        gt=0,
    )

    safety_stock: float = Field(
        default=0,
        ge=0,
    )

    minimum_order_quantity: float = Field(
        default=0,
        ge=0,
    )

    order_multiple: float = Field(
        default=1,
        gt=0,
    )

    scenario_multiplier: float = Field(
        default=1,
        gt=0,
    )

    demand_confidence: float = Field(
        default=1,
        ge=0,
        le=1,
    )

    high_risk: bool = False


@router.get(
    "/health"
)
def health():

    return {
        "status": "ok",
        "service": (
            "autonomous_commercial_ml"
        ),
        "version": "part-30",
    }


@router.post(
    "/decision"
)
def decision(
    request: AutonomousDecisionRequest,
):

    internal = AutonomousRequest(
        outlet_id=request.outlet_id,
        product_id=request.product_id,
        forecast_demand=(
            request.forecast_demand
        ),
        current_inventory=(
            request.current_inventory
        ),
        lead_time_days=(
            request.lead_time_days
        ),
        safety_stock=(
            request.safety_stock
        ),
        minimum_order_quantity=(
            request.minimum_order_quantity
        ),
        order_multiple=(
            request.order_multiple
        ),
        scenario_multiplier=(
            request.scenario_multiplier
        ),
        demand_confidence=(
            request.demand_confidence
        ),
        high_risk=request.high_risk,
    )

    return service.decide_dict(
        internal
    )