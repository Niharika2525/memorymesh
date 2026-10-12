"""Tests for the health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_check_returns_200():
    """Verify that GET /health returns HTTP 200 with expected schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "MemoryMesh"
    assert "version" in data
    assert "environment" in data


def test_lyzr_health_endpoint():
    """Verify that GET /health/lyzr returns HTTP 200 with status schema."""
    response = client.get("/health/lyzr")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert "has_api_key" in data
    assert "has_agent_id" in data
    assert "status" in data


def test_lyzr_remote_api_call_unit(monkeypatch):
    """Verify that LLMService calls Lyzr Agent Studio inference API when configured."""
    from backend.services.llm import LLMService

    service = LLMService(
        lyzr_api_key="mock_lyzr_key",
        lyzr_agent_id="mock_agent_12345",
    )
    assert service.check_lyzr_status()["configured"] is True
    assert service.check_lyzr_status()["has_agent_id"] is True

    # Mock httpx response
    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "Lyzr generated agent plan and reasoning."}

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, headers=None, json=None):
            assert "x-api-key" in headers
            assert headers["x-api-key"] == "mock_lyzr_key"
            assert json["agent_id"] == "mock_agent_12345"
            assert json["message"] == "Execute task"
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)
    res = service.generate_text_lyzr("Execute task")
    assert res == "Lyzr generated agent plan and reasoning."


def test_lyzr_connection_test_endpoint():
    """Verify that GET /health/lyzr/test returns connection diagnostic structure."""
    response = client.get("/health/lyzr/test")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert "status_code" in data
    assert "elapsed_seconds" in data
    assert "success" in data


