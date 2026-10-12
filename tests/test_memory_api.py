"""Tests for MemoryMesh FastAPI memory API endpoints."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.main import app
from backend.memory.qdrant_client import QdrantMemoryService, get_qdrant_service


@pytest.fixture
def override_memory_service():
    """Provide an in-memory Qdrant service dependency override for API tests."""
    in_memory_service = QdrantMemoryService(
        url=":memory:",
        collection_name="api_test_memories",
        vector_size=4,
        distance="Cosine",
        client=QdrantClient(":memory:"),
    )
    app.dependency_overrides[get_qdrant_service] = lambda: in_memory_service
    yield in_memory_service
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_memory_service):
    """Test client with overridden memory service."""
    return TestClient(app)


def test_qdrant_health_endpoint(client):
    """Verify GET /health/qdrant returns connection details."""
    response = client.get("/health/qdrant")
    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is True
    assert data["status"] == "connected"


def test_api_memory_health_endpoint(client):
    """Verify GET /api/memory/health returns memory health status."""
    response = client.get("/api/memory/health")
    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is True


def test_api_store_and_search_memory_flow(client):
    """Verify end-to-end memory store, search, retrieve and delete via API."""
    # 1. Store memory
    store_payload = {
        "id": "api-test-1",
        "content": "Hackathon deadline is tomorrow at 5 PM",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "metadata": {
            "source": "omi_voice",
            "priority": "critical",
        },
    }
    store_res = client.post("/api/memory/store", json=store_payload)
    assert store_res.status_code == 200
    assert store_res.json()["status"] == "stored"
    assert store_res.json()["id"] == "api-test-1"

    # 2. Retrieve by ID
    get_res = client.get("/api/memory/api-test-1")
    assert get_res.status_code == 200
    assert get_res.json()["content"] == "Hackathon deadline is tomorrow at 5 PM"
    assert get_res.json()["metadata"]["priority"] == "critical"

    # 3. Search memories
    search_payload = {
        "vector": [0.95, 0.05, 0.0, 0.0],
        "limit": 3,
    }
    search_res = client.post("/api/memory/search", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["count"] == 1
    assert len(search_data["memories"]) == 1
    assert "Hackathon deadline" in search_data["memories"][0]["content"]
    assert "Relevant Retrieved Context" in search_data["context_text"]

    # 4. Delete memory
    del_res = client.delete("/api/memory/api-test-1")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # 5. Verify deleted
    not_found_res = client.get("/api/memory/api-test-1")
    assert not_found_res.status_code == 404
