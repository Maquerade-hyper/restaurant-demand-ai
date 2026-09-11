from pydantic import BaseModel, Field


class Outlet(BaseModel):
    outlet_id: str
    outlet_type: str
    country: str
    region: str
    city: str
    latitude: float
    longitude: float
    cuisine: str | None = None
    capacity: int | None = Field(default=None, ge=0)
    delivery_available: bool = False
    takeaway_available: bool = False
    bar_available: bool = False