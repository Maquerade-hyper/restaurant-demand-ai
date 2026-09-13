from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.daily_intelligence import DEFAULT_OUTPUT


client = TestClient(app)


@pytest.fixture(scope="module")
def report():
    if not DEFAULT_OUTPUT.exists():
        pytest.skip("Daily intelligence report does not exist.")

    with DEFAULT_OUTPUT.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_health():
    response = client.get("/api/v1/daily-intelligence/health")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == "Restaurant Demand AI"
    assert body["component"] == "Daily Commercial Intelligence API"
    assert body["status"] in {"ready", "waiting"}


def test_summary_is_compact():
    response = client.get("/api/v1/daily-intelligence")

    assert response.status_code == 200

    body = response.json()

    assert "report_date" in body
    assert "actions_required" in body
    assert "priority_summary" in body
    assert "top_actions" in body

    assert len(body["top_actions"]) <= 10


def test_summary_does_not_dump_all_actions(report):
    response = client.get("/api/v1/daily-intelligence")

    assert response.status_code == 200

    body = response.json()

    total_actions = len(report["actions"])

    assert total_actions > 10
    assert len(body["top_actions"]) < total_actions


def test_summary_top_n():
    response = client.get(
        "/api/v1/daily-intelligence?top_n=3"
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["top_actions"]) <= 3


def test_priority_summary():
    response = client.get(
        "/api/v1/daily-intelligence?priority=HIGH"
    )

    assert response.status_code == 200

    body = response.json()

    for action in body["top_actions"]:
        assert action["priority"] == "HIGH"


def test_outlet_filter():
    response = client.get(
        "/api/v1/daily-intelligence?outlet_id=O121"
    )

    assert response.status_code == 200

    body = response.json()

    for action in body["top_actions"]:
        assert action["outlet"]["outlet_id"] == "O121"


def test_product_filter():
    response = client.get(
        "/api/v1/daily-intelligence?product_id=P007"
    )

    assert response.status_code == 200

    body = response.json()

    for action in body["top_actions"]:
        assert action["product"]["product_id"] == "P007"


def test_wrong_date_returns_404(report):
    response = client.get(
        "/api/v1/daily-intelligence?report_date=1900-01-01"
    )

    assert response.status_code == 404


def test_paginated_actions():
    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?page=1&page_size=20"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["total_actions"] > 0
    assert len(body["actions"]) <= 20


def test_pagination_page_two():
    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?page=2&page_size=20"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["page"] == 2
    assert len(body["actions"]) <= 20


def test_paginated_priority_filter():
    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?priority=HIGH&page=1&page_size=10"
    )

    assert response.status_code == 200

    body = response.json()

    for action in body["actions"]:
        assert action["priority"] == "HIGH"


def test_single_action():
    response = client.get(
        "/api/v1/daily-intelligence/actions/O121/P007"
    )

    assert response.status_code in {200, 404}

    if response.status_code == 200:
        body = response.json()

        assert body["action"]["outlet"]["outlet_id"] == "O121"
        assert body["action"]["product"]["product_id"] == "P007"


def test_missing_single_action():
    response = client.get(
        "/api/v1/daily-intelligence/actions/"
        "NONEXISTENT/NONEXISTENT"
    )

    assert response.status_code == 404


def test_action_contract():
    response = client.get(
        "/api/v1/daily-intelligence/actions"
        "?page=1&page_size=1"
    )

    assert response.status_code == 200

    body = response.json()

    if not body["actions"]:
        pytest.skip("No actions available.")

    action = body["actions"][0]

    assert "outlet" in action
    assert "product" in action
    assert "date" in action
    assert "forecast" in action
    assert "inventory" in action
    assert "supply" in action
    assert "priority" in action
    assert "action" in action
    assert "reason" in action


def test_json_serializable():
    response = client.get(
        "/api/v1/daily-intelligence"
    )

    assert response.status_code == 200

    json.dumps(response.json())