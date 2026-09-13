from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _payload(days: int = 42):
    start = date(2026, 1, 1)
    sales = []
    for i in range(days):
        sales.append({
            "date": (start + timedelta(days=i)).isoformat(),
            "outlet_id": "O001",
            "product_id": "P001",
            "quantity_sold": 40 + (i % 7),
        })
        sales.append({
            "date": (start + timedelta(days=i)).isoformat(),
            "outlet_id": "O002",
            "product_id": "P002",
            "quantity_sold": 25 + (i % 5),
        })

    return {
        "client_id": "client_test_001",
        "sales": sales,
        "inventory": [
            {
                "outlet_id": "O001",
                "product_id": "P001",
                "current_inventory": 20,
                "lead_time_days": 3,
                "supplier_risk": "LOW",
            },
            {
                "outlet_id": "O002",
                "product_id": "P002",
                "current_inventory": 500,
                "lead_time_days": 2,
                "supplier_risk": "LOW",
            },
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


def test_health():
    response = client.get("/api/v1/client/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_client_json_to_business_response():
    response = client.post("/api/v1/client/intelligence", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_status"] == "CLIENT_PROVIDED_DATA"
    assert body["model"]["feature_contract"] == "Part 9 / 51 features"
    assert body["model"]["production_champion"] is False
    assert body["summary"]["outlets_analyzed"] == 2
    assert body["summary"]["products_analyzed"] == 2
    assert body["actions"]
    action = body["actions"][0]
    assert action["outlet"]["outlet_name"]
    assert action["product"]["product_name"]
    assert "d1" in action["forecast"]
    assert "d3" in action["forecast"]
    assert "d7" in action["forecast"]
    assert "recommended_order" in action["supply"]
    assert "action" in action["recommendation"]
    assert action["confidence_status"] == "NOT_CALIBRATED"


def test_insufficient_history_rejected():
    payload = _payload(days=20)
    response = client.post("/api/v1/client/intelligence", json=payload)
    assert response.status_code == 422
    assert "requires at least" in response.json()["detail"]
