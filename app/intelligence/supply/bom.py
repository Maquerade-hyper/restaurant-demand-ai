from __future__ import annotations

import pandas as pd


DEFAULT_BOM = pd.DataFrame(
    [
        # product_id, supply_product_id, quantity_per_unit, supply_unit
        ("P001", "P001", 1.00, "liters"),       # Milk
        ("P002", "P002", 1.00, "kg"),           # Chicken
        ("P003", "P003", 1.00, "kg"),           # Beef
        ("P004", "P004", 1.00, "kg"),           # Rice
        ("P005", "P005", 1.00, "kg"),           # Flour
        ("P006", "P006", 1.00, "kg"),           # Cheese
        ("P007", "P007", 1.00, "liters"),       # Cooking Oil
        ("P008", "P008", 1.00, "pieces"),       # Eggs
        ("P009", "P009", 1.00, "pieces"),       # Bread
        ("P010", "P010", 1.00, "pieces"),       # Pizza Dough
        ("P011", "P011", 1.00, "pieces"),       # Pizza Base 12
        ("P012", "P012", 1.00, "pieces"),       # Pizza Base 14
        ("P013", "P013", 1.00, "pieces"),       # Pizza Base 16
        ("P014", "P014", 1.00, "bottles"),      # Soft Drink
        ("P015", "P015", 1.00, "bottles"),      # Water
        ("P016", "P016", 1.00, "pieces"),       # Coffee
        ("P017", "P017", 1.00, "pieces"),       # Tea
        ("P018", "P018", 1.00, "kg"),           # Tomatoes
        ("P019", "P019", 1.00, "kg"),           # Potatoes
        ("P020", "P020", 1.00, "pieces"),       # Packaging
    ],
    columns=[
        "product_id",
        "supply_product_id",
        "quantity_per_unit",
        "supply_unit",
    ],
)


def validate_bom(bom: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    required = {
        "product_id",
        "supply_product_id",
        "quantity_per_unit",
        "supply_unit",
    }

    missing = required - set(bom.columns)

    if missing:
        errors.append(f"BOM missing columns: {sorted(missing)}")
        return errors

    if bom.empty:
        errors.append("BOM is empty")

    if bom["product_id"].isna().any():
        errors.append("BOM contains null product_id")

    if bom["supply_product_id"].isna().any():
        errors.append("BOM contains null supply_product_id")

    if (bom["quantity_per_unit"] <= 0).any():
        errors.append("BOM quantity_per_unit must be > 0")

    if bom.duplicated(
        subset=["product_id", "supply_product_id"]
    ).any():
        errors.append(
            "BOM contains duplicate product/supply mappings"
        )

    return errors


def normalize_bom(bom: pd.DataFrame | None = None) -> pd.DataFrame:
    result = (
        DEFAULT_BOM.copy()
        if bom is None
        else bom.copy()
    )

    errors = validate_bom(result)

    if errors:
        raise ValueError("; ".join(errors))

    result["quantity_per_unit"] = pd.to_numeric(
        result["quantity_per_unit"],
        errors="raise",
    )

    result["product_id"] = result["product_id"].astype(str)
    result["supply_product_id"] = (
        result["supply_product_id"].astype(str)
    )

    return result.reset_index(drop=True)


def explode_demand_to_supply(
    demand: pd.DataFrame,
    bom: pd.DataFrame | None = None,
) -> pd.DataFrame:

    required = {
        "outlet_id",
        "product_id",
        "date",
        "demand",
    }

    missing = required - set(demand.columns)

    if missing:
        raise ValueError(
            f"Demand missing columns: {sorted(missing)}"
        )

    mapping = normalize_bom(bom)

    result = demand.copy()

    result["date"] = pd.to_datetime(result["date"])
    result["demand"] = pd.to_numeric(
        result["demand"],
        errors="coerce",
    ).fillna(0.0)

    result = result.merge(
        mapping,
        on="product_id",
        how="left",
        validate="many_to_many",
    )

    result["supply_requirement"] = (
        result["demand"]
        * result["quantity_per_unit"]
    )

    return result[
        [
            "outlet_id",
            "date",
            "product_id",
            "demand",
            "supply_product_id",
            "supply_unit",
            "quantity_per_unit",
            "supply_requirement",
        ]
    ]