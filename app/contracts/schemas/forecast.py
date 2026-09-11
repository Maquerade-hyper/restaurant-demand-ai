from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    outlet_id: str = Field(min_length=1)
    forecast_days: int = Field(default=7, ge=1, le=30)