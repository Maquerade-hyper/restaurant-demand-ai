from datetime import date, timedelta

import pandas as pd

from app.data.generators.behavior import demand_multiplier
from app.data.generators.external_behavior import (
    event_multiplier,
    promotion_multiplier,
    weather_multiplier,
)
from app.data.generators.operational_behavior import (
    calculate_observed_sales,
)


def generate_behavior_sample(days: int = 90) -> pd.DataFrame:
    rows = []

    start = date(2025, 1, 1)

    for i in range(days):
        day = start + timedelta(days=i)

        multiplier = demand_multiplier(
            day=day,
            outlet_type="restaurant",
            category="meat",
            location_type="mixed",
            tourism_level=0.7,
            business_area=0.5,
            residential_area=0.6,
            student_area=0.4,
        )

        base_demand = 100.0

        true_demand = base_demand * multiplier

        operational = calculate_observed_sales(
            true_demand=true_demand,
            available_stock=120.0 + (i % 7) * 25.0,
        )

        rows.append(
            {
                "date": day,
                "true_demand": true_demand,
                "observed_sales": operational["observed_sales"],
                "lost_demand": operational["lost_demand"],
                "stockout": operational["stockout"],
                "multiplier": multiplier,
            }
        )

    return pd.DataFrame(rows)


def validate_behavior(df: pd.DataFrame) -> dict:
    checks = {
        "rows_exist": len(df) > 0,
        "demand_positive": (df["true_demand"] > 0).all(),
        "sales_non_negative": (df["observed_sales"] >= 0).all(),
        "lost_demand_non_negative": (df["lost_demand"] >= 0).all(),
        "sales_not_above_demand": (
            df["observed_sales"] <= df["true_demand"]
        ).all(),
        "stockout_consistent": (
            df["stockout"]
            == (df["lost_demand"] > 0)
        ).all(),
    }

    return checks


def run_validation() -> dict:
    df = generate_behavior_sample()

    checks = validate_behavior(df)

    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "rows": len(df),
        "mean_demand": round(df["true_demand"].mean(), 2),
        "mean_sales": round(df["observed_sales"].mean(), 2),
        "stockout_rate": round(df["stockout"].mean(), 4),
    }


if __name__ == "__main__":
    result = run_validation()

    print("BEHAVIOR VALIDATION")
    print("-" * 40)

    for name, passed in result["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("-" * 40)
    print(f"ROWS: {result['rows']}")
    print(f"MEAN DEMAND: {result['mean_demand']}")
    print(f"MEAN SALES: {result['mean_sales']}")
    print(f"STOCKOUT RATE: {result['stockout_rate']}")
    print(f"ALL PASSED: {result['all_passed']}")