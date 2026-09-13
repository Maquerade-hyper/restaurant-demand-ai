from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intelligence.client_api.service import ClientIntelligenceService


def build_payload():
    start = date(2026, 1, 1)
    sales = []
    for i in range(42):
        sales.append({"date": (start + timedelta(days=i)).isoformat(), "outlet_id": "O001", "product_id": "P001", "quantity_sold": 40 + i % 7})
        sales.append({"date": (start + timedelta(days=i)).isoformat(), "outlet_id": "O002", "product_id": "P002", "quantity_sold": 25 + i % 5})
    return {
        "client_id": "local_api_fixture",
        "sales": sales,
        "inventory": [
            {"outlet_id": "O001", "product_id": "P001", "current_inventory": 20, "lead_time_days": 3, "supplier_risk": "LOW"},
            {"outlet_id": "O002", "product_id": "P002", "current_inventory": 500, "lead_time_days": 2, "supplier_risk": "LOW"},
        ],
        "outlets": [
            {"outlet_id": "O001", "outlet_name": "Main Restaurant", "outlet_type": "restaurant"},
            {"outlet_id": "O002", "outlet_name": "Cloud Kitchen A", "outlet_type": "cloud_kitchen"},
        ],
        "products": [
            {"product_id": "P001", "product_name": "Chicken", "category": "protein", "unit": "kg"},
            {"product_id": "P002", "product_name": "Rice", "category": "grain", "unit": "kg"},
        ],
        "top_n": 10,
        "include_normal": True,
    }


if __name__ == "__main__":
    result = ClientIntelligenceService().generate(build_payload())
    print(json.dumps(result, indent=2, default=str))
    print("CLIENT JSON INTELLIGENCE: PASS")
