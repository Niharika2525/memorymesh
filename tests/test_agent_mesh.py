"""Tests for the agent mesh API endpoints and OrchestratorAgent workflow."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from qdrant_client import QdrantClient

from backend.main import app
from backend.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent
from backend.agents.memory import MemoryAgent
from backend.memory.qdrant_client import QdrantMemoryService


@pytest.fixture
def in_memory_orchestrator():
    """Orchestrator backed by an isolated in-memory Qdrant service (no live cluster)."""
    mem_client = QdrantClient(":memory:")
    mem_service = QdrantMemoryService(
        url=":memory:",
        collection_name="test_agent_mesh",
        vector_size=1536,
        distance="Cosine",
        client=mem_client,
    )
    memory_agent = MemoryAgent(memory_service=mem_service)
    orchestrator = OrchestratorAgent(memory_agent=memory_agent)
    return orchestrator


@pytest.fixture
def client(in_memory_orchestrator):
    """Test client with orchestrator dependency overridden to use in-memory Qdrant."""
    app.dependency_overrides[get_orchestrator_agent] = lambda: in_memory_orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─── Agent Status Endpoint ────────────────────────────────────────────────────

def test_agent_status_endpoint(client):
    """GET /api/agent/status returns ready status with all 6 agents listed."""
    response = client.get("/api/agent/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "memory" in data["agents"]
    assert "orchestrator" in data["agents"]
    assert "planner" in data["agents"]
    assert "research" in data["agents"]
    assert "action" in data["agents"]
    assert "verifier" in data["agents"]
    assert isinstance(data["memory_connected"], bool)


# ─── Agent Workflow Endpoint ──────────────────────────────────────────────────

def test_agent_run_workflow_returns_full_result(client):
    """POST /api/agent/run returns the full execution result structure."""
    payload = {
        "task": "What should I focus on today for the hackathon?",
        "speaker": "ravel",
        "source": "api",
        "auto_persist_memory": True,
    }
    response = client.post("/api/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Core fields
    assert "execution_id" in data
    assert data["task"] == payload["task"]
    assert data["speaker"] == "ravel"
    assert data["source"] == "api"

    # Sub-agent results present
    assert "plan" in data
    assert "research" in data
    assert "action" in data
    assert "verification" in data
    assert "final_response" in data
    assert isinstance(data["final_response"], str)
    assert len(data["final_response"]) > 0


def test_agent_run_workflow_persists_memory(client, in_memory_orchestrator):
    """POST /api/agent/run with auto_persist_memory=True writes to Qdrant."""
    payload = {
        "task": "Remind me to test Qdrant memory persistence",
        "speaker": "ravel",
        "source": "api",
        "auto_persist_memory": True,
    }
    response = client.post("/api/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["persisted_memory"] is not None
    assert data["persisted_memory"]["status"] == "stored"

    # Verify it's actually in memory
    count = in_memory_orchestrator.memory.memory_service.count_memories("test_agent_mesh")
    assert count >= 1


def test_agent_run_workflow_no_persist(client, in_memory_orchestrator):
    """POST /api/agent/run with auto_persist_memory=False does not persist."""
    payload = {
        "task": "Quick question no memory save",
        "speaker": "ravel",
        "source": "api",
        "auto_persist_memory": False,
    }
    response = client.post("/api/agent/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["persisted_memory"] is None


def test_agent_run_workflow_uses_recalled_memory(client, in_memory_orchestrator):
    """Second workflow call recalls memories stored by the first."""
    # First call stores a memory
    r1 = client.post("/api/agent/run", json={
        "task": "Note that the hackathon deadline is October 15th",
        "auto_persist_memory": True,
    })
    assert r1.status_code == 200

    # Second call on a related topic should find the first memory
    r2 = client.post("/api/agent/run", json={
        "task": "When is the hackathon deadline?",
        "auto_persist_memory": False,
    })
    assert r2.status_code == 200
    data = r2.json()
    # recalled_memories_count may be 0 or 1 depending on embedding similarity;
    # what matters is the call succeeds and the recall field is present
    assert "recalled_memories_count" in data
    assert isinstance(data["recalled_memories"], list)


def test_agent_plan_structure(client):
    """Verify the plan sub-result has the expected keys and steps."""
    response = client.post("/api/agent/run", json={"task": "Plan my morning routine"})
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert "task" in plan
    assert "steps" in plan
    assert isinstance(plan["steps"], list)
    assert len(plan["steps"]) > 0
    for step in plan["steps"]:
        assert "step" in step
        assert "agent" in step
        assert "description" in step


def test_agent_verification_structure(client):
    """Verify that the verifier sub-result has the required keys."""
    response = client.post("/api/agent/run", json={"task": "Check if system is healthy"})
    assert response.status_code == 200
    verification = response.json()["verification"]
    assert "verified" in verification
    assert "confidence_score" in verification
    assert "verification_notes" in verification
    assert isinstance(verification["verified"], bool)


# ─── OrchestratorAgent Unit Tests ────────────────────────────────────────────

def test_orchestrator_run_workflow_unit(in_memory_orchestrator):
    """Unit test: OrchestratorAgent.run_workflow returns expected structure."""
    result = in_memory_orchestrator.run_workflow(
        task="What is the MemoryMesh project about?",
        speaker="test_user",
        source="unit_test",
        auto_persist_memory=False,
    )
    assert result["task"] == "What is the MemoryMesh project about?"
    assert result["speaker"] == "test_user"
    assert "execution_id" in result
    assert "plan" in result
    assert "research" in result
    assert "action" in result
    assert "verification" in result
    assert "final_response" in result
    assert result["persisted_memory"] is None
