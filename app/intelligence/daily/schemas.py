from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class OutletIdentity(BaseModel):
    outlet_id: str
    outlet_name: str
    outlet_type: str
    country: str
    region: str
    city: str
    location_type: str
    cuisine: str | None = None
    capacity: int | None = None
    kitchen_type: str | None = None
    delivery_available: bool | None = None
    takeaway_available: bool | None = None
    bar_available: bool | None = None


class ProductIdentity(BaseModel):
    product_id: str
    product_name: str
    category: str
    unit: str


class ForecastSummary(BaseModel):
    d1: float = Field(ge=0)
    d3: float = Field(ge=0)
    d7: float = Field(ge=0)


class InventorySummary(BaseModel):
    current_inventory: float = Field(ge=0)
    required_inventory: float = Field(ge=0)
    shortage: float = Field(ge=0)
    safety_stock: float = Field(ge=0)


class SupplySummary(BaseModel):
    lead_time_days: int = Field(ge=0)
    lead_time_demand: float = Field(ge=0)
    recommended_order: float = Field(ge=0)
    order_unit: str
    order_multiple: float = Field(gt=0)
    supplier_risk: str


class RecommendationSummary(BaseModel):
    action: str
    priority: str
    reason: str


class DailyAction(BaseModel):
    outlet: OutletIdentity
    product: ProductIdentity

    date: str

    forecast: ForecastSummary
    demand_risk: str
    demand_spike: str
    model_confidence: float = Field(ge=0, le=1)

    inventory: InventorySummary
    supply: SupplySummary

    stockout_risk: str
    priority: str

    recommendation: RecommendationSummary

    key_drivers: List[str]


class DailySummary(BaseModel):
    outlets_analyzed: int = Field(ge=0)
    products_analyzed: int = Field(ge=0)
    outlet_product_series_analyzed: int = Field(ge=0)

    actions_required: int = Field(ge=0)

    critical: int = Field(ge=0)
    high: int = Field(ge=0)
    moderate: int = Field(ge=0)
    low: int = Field(ge=0)
    normal: int = Field(ge=0)

    replenish_actions: int = Field(ge=0)
    review_actions: int = Field(ge=0)
    hold_actions: int = Field(ge=0)


class DailyIntelligenceResponse(BaseModel):
    service: str
    version: str

    report_date: str

    source_status: str
    generated_at: str

    summary: DailySummary

    actions: List[DailyAction]

    validation: dict