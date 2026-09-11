from pydantic import BaseModel


class Product(BaseModel):
    product_id: str
    product_name: str
    category: str
    unit: str
    unit_size: float | None = None
    dimension: str | None = None
    pack_size: float | None = None
    shelf_life_days: int | None = None