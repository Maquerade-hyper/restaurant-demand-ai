from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


OUTLET_CANDIDATES = [
    ROOT / "data" / "synthetic" / "outlets.csv",
    ROOT / "data" / "processed" / "outlets.csv",
    ROOT / "data" / "raw" / "outlets.csv",
]


PRODUCT_CANDIDATES = [
    ROOT / "data" / "synthetic" / "products.csv",
    ROOT / "data" / "processed" / "products.csv",
    ROOT / "data" / "raw" / "products.csv",
]


PRODUCT_CATALOG: Dict[str, Dict[str, str]] = {
    "P001": {"name": "Milk", "category": "Dairy", "unit": "litre"},
    "P002": {"name": "Chicken", "category": "Meat", "unit": "kg"},
    "P003": {"name": "Beef", "category": "Meat", "unit": "kg"},
    "P004": {"name": "Rice", "category": "Staples", "unit": "kg"},
    "P005": {"name": "Flour", "category": "Staples", "unit": "kg"},
    "P006": {"name": "Cheese", "category": "Dairy", "unit": "kg"},
    "P007": {"name": "Cooking Oil", "category": "Staples", "unit": "litre"},
    "P008": {"name": "Eggs", "category": "Dairy", "unit": "pieces"},
    "P009": {"name": "Bread", "category": "Bakery", "unit": "pieces"},
    "P010": {"name": "Pizza Dough", "category": "Pizza", "unit": "pieces"},
    "P011": {
        "name": "Pizza Base 12-inch",
        "category": "Pizza",
        "unit": "pieces",
    },
    "P012": {
        "name": "Pizza Base 14-inch",
        "category": "Pizza",
        "unit": "pieces",
    },
    "P013": {
        "name": "Pizza Base 16-inch",
        "category": "Pizza",
        "unit": "pieces",
    },
    "P014": {
        "name": "Soft Drink",
        "category": "Beverage",
        "unit": "bottles",
    },
    "P015": {
        "name": "Bottled Water",
        "category": "Beverage",
        "unit": "bottles",
    },
    "P016": {"name": "Coffee", "category": "Beverage", "unit": "kg"},
    "P017": {"name": "Tea", "category": "Beverage", "unit": "kg"},
    "P018": {
        "name": "Tomatoes",
        "category": "Produce",
        "unit": "kg",
    },
    "P019": {
        "name": "Potatoes",
        "category": "Produce",
        "unit": "kg",
    },
    "P020": {
        "name": "Packaging Box",
        "category": "Packaging",
        "unit": "pieces",
    },
}


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_outlet_catalog() -> pd.DataFrame:
    path = _first_existing(OUTLET_CANDIDATES)

    if path is None:
        raise FileNotFoundError(
            "Outlet catalog not found. Expected one of: "
            + ", ".join(str(p) for p in OUTLET_CANDIDATES)
        )

    df = pd.read_csv(path)

    required = {
        "outlet_id",
        "outlet_type",
        "country",
        "region",
        "city",
        "location_type",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Outlet catalog missing required columns: {sorted(missing)}"
        )

    df["outlet_id"] = df["outlet_id"].astype(str)

    return df.drop_duplicates("outlet_id").reset_index(drop=True)


def load_product_catalog() -> pd.DataFrame:
    path = _first_existing(PRODUCT_CANDIDATES)

    if path is not None:
        df = pd.read_csv(path)

        if "product_id" in df.columns:
            df["product_id"] = df["product_id"].astype(str)

            result = pd.DataFrame(
                {
                    "product_id": df["product_id"],
                    "product_name": df.get(
                        "product_name",
                        df.get("name", df["product_id"]),
                    ),
                    "category": df.get(
                        "category",
                        "Unknown",
                    ),
                    "unit": df.get(
                        "unit",
                        "units",
                    ),
                }
            )

            return result.drop_duplicates("product_id")

    rows = []

    for product_id, info in PRODUCT_CATALOG.items():
        rows.append(
            {
                "product_id": product_id,
                "product_name": info["name"],
                "category": info["category"],
                "unit": info["unit"],
            }
        )

    return pd.DataFrame(rows)


def build_outlet_identity(
    outlet_id: str,
    catalog: pd.DataFrame,
) -> dict:
    matches = catalog[
        catalog["outlet_id"].astype(str) == str(outlet_id)
    ]

    if matches.empty:
        raise ValueError(
            f"Outlet {outlet_id} does not exist in the outlet catalog."
        )

    row = matches.iloc[0]

    def value(column: str, default=None):
        if column not in row.index:
            return default

        raw = row[column]

        if pd.isna(raw):
            return default

        return raw

    return {
        "outlet_id": str(value("outlet_id")),
        "outlet_name": f"Outlet {value('outlet_id')}",
        "outlet_type": str(value("outlet_type", "unknown")),
        "country": str(value("country", "unknown")),
        "region": str(value("region", "unknown")),
        "city": str(value("city", "unknown")),
        "location_type": str(value("location_type", "unknown")),
        "cuisine": value("cuisine"),
        "capacity": (
            int(value("capacity"))
            if value("capacity") is not None
            else None
        ),
        "kitchen_type": value("kitchen_type"),
        "delivery_available": value("delivery_available"),
        "takeaway_available": value("takeaway_available"),
        "bar_available": value("bar_available"),
    }


def build_product_identity(
    product_id: str,
    catalog: pd.DataFrame,
) -> dict:
    matches = catalog[
        catalog["product_id"].astype(str) == str(product_id)
    ]

    if not matches.empty:
        row = matches.iloc[0]

        return {
            "product_id": str(product_id),
            "product_name": str(row["product_name"]),
            "category": str(row["category"]),
            "unit": str(row["unit"]),
        }

    fallback = PRODUCT_CATALOG.get(str(product_id))

    if fallback is None:
        return {
            "product_id": str(product_id),
            "product_name": str(product_id),
            "category": "Unknown",
            "unit": "units",
        }

    return {
        "product_id": str(product_id),
        "product_name": fallback["name"],
        "category": fallback["category"],
        "unit": fallback["unit"],
    }