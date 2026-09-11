from pathlib import Path

import pandas as pd

from app.data.schemas.outlet_schema import OutletRecord
from app.data.schemas.product_schema import ProductRecord
from app.data.schemas.sales_schema import SalesRecord
from app.data.schemas.inventory_schema import InventoryRecord
from app.data.schemas.external_schema import (
    HolidayRecord,
    EventRecord,
    WeatherRecord,
    DemographicRecord,
    PromotionRecord,
)


DATA_DIR = Path("data/synthetic")


def validate_dataframe(df: pd.DataFrame, model) -> list[str]:
    errors = []

    # Convert pandas NaN/NaT to Python None
    clean_df = df.astype(object).where(
        pd.notna(df),
        None,
    )

    for index, row in clean_df.iterrows():
        try:
            model.model_validate(row.to_dict())
        except Exception as exc:
            errors.append(
                f"row={index}: {exc}"
            )

    return errors


def validate_outlets():
    return validate_dataframe(
        pd.read_csv(DATA_DIR / "outlets.csv"),
        OutletRecord,
    )


def validate_products():
    return validate_dataframe(
        pd.read_csv(DATA_DIR / "products.csv"),
        ProductRecord,
    )


def validate_sales():
    path = DATA_DIR / "sales.csv"

    if not path.exists():
        return ["sales.csv does not exist"]

    return validate_dataframe(
        pd.read_csv(path),
        SalesRecord,
    )


def validate_inventory():
    path = DATA_DIR / "inventory.csv"

    if not path.exists():
        return ["inventory.csv does not exist"]

    return validate_dataframe(
        pd.read_csv(path),
        InventoryRecord,
    )


def validate_external():
    errors = []

    files = [
        ("holidays.csv", HolidayRecord),
        ("events.csv", EventRecord),
        ("weather.csv", WeatherRecord),
        ("demographics.csv", DemographicRecord),
        ("promotions.csv", PromotionRecord),
    ]

    for filename, model in files:
        path = DATA_DIR / filename

        if path.exists():
            errors.extend(
                validate_dataframe(
                    pd.read_csv(path),
                    model,
                )
            )

    return errors