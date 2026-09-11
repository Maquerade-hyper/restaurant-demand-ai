from datetime import date

from pydantic import BaseModel, Field


class HolidayRecord(BaseModel):
    date: date
    country: str
    region: str | None = None
    holiday_name: str
    holiday_type: str
    importance: float = Field(ge=0)


class EventRecord(BaseModel):
    event_id: str
    event_name: str
    event_type: str
    city: str

    latitude: float
    longitude: float

    start_date: date
    end_date: date
    importance: float = Field(ge=0)


class WeatherRecord(BaseModel):
    date: date
    city: str

    temperature: float
    precipitation: float = Field(ge=0)
    humidity: float = Field(ge=0, le=100)

    weather_condition: str


class DemographicRecord(BaseModel):
    city: str

    population: int = Field(ge=0)
    population_density: float = Field(ge=0)

    tourism_index: float = Field(ge=0)
    student_index: float = Field(ge=0)
    business_index: float = Field(ge=0)


class PromotionRecord(BaseModel):
    outlet_id: str
    product_id: str

    start_date: date
    end_date: date

    promotion_type: str
    discount: float = Field(ge=0, le=100)