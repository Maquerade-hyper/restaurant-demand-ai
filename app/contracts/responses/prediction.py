from pydantic import BaseModel


class ProductForecast(BaseModel):
    product_id: str
    product_name: str
    unit: str
    predicted_demand: float
    historical_baseline: float
    change_amount: float
    change_percent: float
    direction: str
    risk: str


class ForecastResponse(BaseModel):
    outlet_id: str
    forecast_days: int
    forecasts: list[ProductForecast]