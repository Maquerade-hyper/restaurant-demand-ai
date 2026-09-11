from pathlib import Path

import pandas as pd

from app.data.validation import (
    validate_outlets,
    validate_products,
)
from app.data.quality import (
    check_no_nulls,
    check_unique,
    check_non_negative,
)
from app.data.time_validation import (
    validate_dates,
)


DATA_DIR = Path("data/synthetic")


def validate_dataset() -> dict:

    errors = []

    outlets = pd.read_csv(
        DATA_DIR / "outlets.csv"
    )

    products = pd.read_csv(
        DATA_DIR / "products.csv"
    )

    # Schema
    errors.extend(validate_outlets())
    errors.extend(validate_products())

    # Outlet quality
    errors.extend(
        check_no_nulls(
            outlets,
            [
                "outlet_id",
                "outlet_type",
                "country",
                "region",
                "city",
            ],
        )
    )

    errors.extend(
        check_unique(
            outlets,
            "outlet_id",
        )
    )

    errors.extend(
        check_non_negative(
            outlets,
            [
                "capacity",
                "tourism_level",
                "business_area",
                "residential_area",
                "student_area",
            ],
        )
    )

    # Product quality
    errors.extend(
        check_no_nulls(
            products,
            [
                "product_id",
                "product_name",
                "category",
                "unit",
            ],
        )
    )

    errors.extend(
        check_unique(
            products,
            "product_id",
        )
    )

    # Coordinate validation
    if not (
        outlets["latitude"].between(-90, 90).all()
    ):
        errors.append(
            "Invalid latitude values"
        )

    if not (
        outlets["longitude"].between(-180, 180).all()
    ):
        errors.append(
            "Invalid longitude values"
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "outlets": len(outlets),
        "products": len(products),
    }


if __name__ == "__main__":

    result = validate_dataset()

    print("DATASET VALIDATION")
    print("-" * 40)

    print(f"OUTLETS: {result['outlets']}")
    print(f"PRODUCTS: {result['products']}")

    if result["passed"]:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

        for error in result["errors"]:
            print(f"- {error}")