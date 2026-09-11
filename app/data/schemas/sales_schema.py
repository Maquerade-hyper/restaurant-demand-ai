from datetime import date

from pydantic import BaseModel, Field


class SalesRecord(BaseModel):
    outlet_id: str
    date: date
    product_id: str

    quantity_sold: float = Field(ge=0)
    unit: str

    revenue: float = Field(ge=0)