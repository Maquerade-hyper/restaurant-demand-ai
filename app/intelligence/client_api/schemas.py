from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class ClientSalesRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: date
    outlet_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    quantity_sold: float = Field(ge=0)


class ClientInventoryRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    outlet_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    current_inventory: float = Field(ge=0)
    lead_time_days: int = Field(default=1, ge=1, le=90)
    safety_stock: float | None = Field(default=None, ge=0)
    minimum_order_quantity: float | None = Field(default=None, ge=0)
    order_multiple: float | None = Field(default=None, gt=0)
    supplier_risk: str | None = None


class ClientOutlet(BaseModel):
    model_config = ConfigDict(extra="allow")

    outlet_id: str = Field(min_length=1)
    outlet_name: str | None = None
    outlet_type: str | None = None
    country: str | None = None
    region: str | None = None
    city: str | None = None
    location_type: str | None = None


class ClientProduct(BaseModel):
    model_config = ConfigDict(extra="allow")

    product_id: str = Field(min_length=1)
    product_name: str | None = None
    category: str | None = None
    unit: str | None = None


class ClientIntelligenceRequest(BaseModel):
    """Canonical JSON contract for client applications."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1)
    as_of_date: date | None = None
    sales: list[ClientSalesRow] = Field(min_length=1)
    inventory: list[ClientInventoryRow] = Field(default_factory=list)
    outlets: list[ClientOutlet] = Field(default_factory=list)
    products: list[ClientProduct] = Field(default_factory=list)
    top_n: int = Field(default=50, ge=1, le=500)
    include_normal: bool = False


class ClientIntelligenceResponse(BaseModel):
    service: str
    version: str
    request_id: str
    client_id: str
    report_date: date
    source_status: str
    model: dict[str, Any]
    summary: dict[str, Any]
    actions: list[dict[str, Any]]
    data_quality: dict[str, Any]
