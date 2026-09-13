from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.main import app


client = TestClient(app)


def check(name: str, condition: bool) -> None:
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        raise SystemExit(1)


def main() -> None:

    print()
    print("=" * 70)
    print("RESTAURANT DEMAND AI - DAILY INTELLIGENCE API")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. HEALTH
    # ---------------------------------------------------------

    print()
    print("1. API HEALTH")

    response = client.get(
        "/api/v1/daily-intelligence/health"
    )

    check("API health", response.status_code == 200)

    health = response.json()

    print("Status :", response.status_code)
    print("Report :", health.get("report_date"))
    print("Ready  :", health.get("status"))

    # ---------------------------------------------------------
    # 2. COMPACT SUMMARY
    # ---------------------------------------------------------

    print()
    print("2. COMPACT DAILY SUMMARY")

    response = client.get(
        "/api/v1/daily-intelligence"
    )

    check("Summary endpoint", response.status_code == 200)

    summary = response.json()

    print("Report date      :", summary["report_date"])
    print("Outlets analyzed :", summary["outlets_analyzed"])
    print("Products analyzed:", summary["products_analyzed"])
    print("Actions required :", summary["actions_required"])
    print("Top actions      :", len(summary["top_actions"]))

    check(
        "Response is compact",
        len(summary["top_actions"]) <= 10,
    )

    # ---------------------------------------------------------
    # 3. PAGINATION
    # ---------------------------------------------------------

    print()
    print("3. PAGINATED ACTIONS")

    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?page=1&page_size=20"
    )

    check(
        "Paginated endpoint",
        response.status_code == 200,
    )

    page = response.json()

    print("Page           :", page["page"])
    print("Page size      :", page["page_size"])
    print("Total actions  :", page["total_actions"])
    print("Returned       :", len(page["actions"]))
    print("Total pages    :", page["total_pages"])

    check(
        "Page <= requested size",
        len(page["actions"]) <= 20,
    )

    # ---------------------------------------------------------
    # 4. HIGH PRIORITY
    # ---------------------------------------------------------

    print()
    print("4. HIGH PRIORITY FILTER")

    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?priority=HIGH&page=1&page_size=20"
    )

    check(
        "High-priority filter",
        response.status_code == 200,
    )

    high = response.json()

    print("High-priority total:", high["total_actions"])
    print("Returned           :", len(high["actions"]))

    check(
        "All returned actions are HIGH",
        all(
            item["priority"] == "HIGH"
            for item in high["actions"]
        ),
    )

    # ---------------------------------------------------------
    # 5. OUTLET
    # ---------------------------------------------------------

    print()
    print("5. OUTLET FILTER")

    response = client.get(
        "/api/v1/daily-intelligence"
        "?outlet_id=O121"
    )

    check(
        "Outlet filter",
        response.status_code == 200,
    )

    outlet = response.json()

    print(
        "Outlet O121 actions:",
        outlet["actions_required"],
    )

    # ---------------------------------------------------------
    # 6. PRODUCT
    # ---------------------------------------------------------

    print()
    print("6. PRODUCT FILTER")

    response = client.get(
        "/api/v1/daily-intelligence"
        "?product_id=P007"
    )

    check(
        "Product filter",
        response.status_code == 200,
    )

    product = response.json()

    print(
        "Product P007 actions:",
        product["actions_required"],
    )

    # ---------------------------------------------------------
    # 7. SINGLE ACTION
    # ---------------------------------------------------------

    print()
    print("7. SINGLE OUTLET/PRODUCT ACTION")

    response = client.get(
        "/api/v1/daily-intelligence/actions/O121/P007"
    )

    check(
        "Single action endpoint",
        response.status_code in {200, 404},
    )

    if response.status_code == 200:
        action = response.json()["action"]

        print(
            "Outlet :",
            action["outlet"]["outlet_name"],
        )

        print(
            "Product:",
            action["product"]["product_name"],
        )

        print(
            "Action :",
            action["action"],
        )

        print(
            "Order  :",
            action["supply"]["recommended_order"],
            action["supply"]["order_unit"],
        )

    # ---------------------------------------------------------
    # 8. JSON SERIALIZATION
    # ---------------------------------------------------------

    print()
    print("8. JSON SERIALIZATION")

    response = client.get(
        "/api/v1/daily-intelligence"
    )

    try:
        json.dumps(response.json())
        serializable = True
    except Exception:
        serializable = False

    check(
        "JSON serialization",
        serializable,
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("DAILY INTELLIGENCE API: PASS")
    print("=" * 70)
    print()
    print("The API now returns:")
    print("  • compact summary")
    print("  • top actions")
    print("  • paginated actions")
    print("  • outlet/product filtering")
    print("  • single-action lookup")
    print()
    print("Ready for cloud deployment testing.")
    print()


if __name__ == "__main__":
    main()