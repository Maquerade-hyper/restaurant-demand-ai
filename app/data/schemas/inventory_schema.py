from datetime import date

from pydantic import BaseModel, Field


class InventoryRecord(BaseModel):
    outlet_id: str
    date: date
    product_id: str

    opening_stock: float = Field(ge=0)
    received_stock: float = Field(ge=0)
    closing_stock: float = Field(ge=0)

    wastage: float = Field(default=0, ge=0)
    stockout: bool = False