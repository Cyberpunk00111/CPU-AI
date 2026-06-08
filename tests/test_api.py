"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.main import create_app


def test_health_endpoint() -> None:
    """Health endpoint should report liveness."""
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_query_endpoint_accepts_message() -> None:
    """Query endpoint should start a stream job."""
    client = TestClient(create_app())
    response = client.post(
        "/api/query",
        json={"message": "What is CogniCore?", "use_web_search": False, "max_tokens": 8},
    )
    assert response.status_code == 200
    assert response.json()["accepted"] is True
