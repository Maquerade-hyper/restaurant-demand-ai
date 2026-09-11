from pydantic import BaseModel, Field


class OutletRecord(BaseModel):
    outlet_id: str
    outlet_type: str

    country: str
    region: str
    city: str

    latitude: float
    longitude: float

    location_type: str
    cuisine: str | None = None

    capacity: int | None = Field(default=None, ge=0)

    delivery_available: bool = False
    takeaway_available: bool = False
    bar_available: bool = False

    kitchen_type: str | None = None

    tourism_level: float = 0.0
    business_area: float = 0.0
    residential_area: float = 0.0
    student_area: float = 0.0