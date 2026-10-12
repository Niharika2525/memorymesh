"""Tests for Omi voice listener endpoints and OmiVoiceListener service."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.main import app
from backend.omi.listener import OmiVoiceListener, get_omi_listener
from backend.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent
from backend.agents.memory import MemoryAgent
from backend.memory.qdrant_client import QdrantMemoryService


@pytest.fixture
def in_memory_orchestrator():
    """Orchestrator with isolated in-memory Qdrant for Omi tests."""
    mem_service = QdrantMemoryService(
        url=":memory:",
        collection_name="test_omi_memories",
        vector_size=1536,
        distance="Cosine",
        client=QdrantClient(":memory:"),
    )
    return OrchestratorAgent(memory_agent=MemoryAgent(memory_service=mem_service))


@pytest.fixture
def client(in_memory_orchestrator):
    """Test client with both Omi listener and orchestrator overridden."""
    omi_listener = OmiVoiceListener(orchestrator=in_memory_orchestrator)
    app.dependency_overrides[get_omi_listener] = lambda: omi_listener
    app.dependency_overrides[get_orchestrator_agent] = lambda: in_memory_orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─── Omi Status Endpoint ──────────────────────────────────────────────────────

def test_omi_status_endpoint(client):
    """GET /api/omi/status returns active service status."""
    response = client.get("/api/omi/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["service"] == "OmiVoiceListener"
    assert "has_app_id" in data


# ─── Omi Voice Endpoint ───────────────────────────────────────────────────────

def test_omi_voice_endpoint_returns_response(client):
    """POST /api/omi/voice runs agent mesh and returns speech_response."""
    payload = {
        "transcript": "Set a reminder to review the Qdrant integration tomorrow morning",
        "speaker": "ravel",
        "session_id": "session-001",
    }
    response = client.post("/api/omi/voice", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["source"] == "omi_voice"
    assert data["transcript"] == payload["transcript"]
    assert "speech_response" in data
    assert data["speech_response"] is not None
    assert "orchestrator_result" in data


def test_omi_voice_empty_transcript_is_ignored(client):
    """POST /api/omi/voice with empty transcript returns ignored status."""
    response = client.post("/api/omi/voice", json={"transcript": "   "})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"


# ─── Omi Webhook Endpoint ─────────────────────────────────────────────────────

def test_omi_webhook_standard_payload(client):
    """POST /api/omi/webhook handles standard Omi webhook transcript event."""
    payload = {
        "event": "transcript",
        "data": {
            "transcript": "What meetings do I have this week?",
            "speaker": "ravel",
            "session_id": "webhook-session-42",
        },
    }
    response = client.post("/api/omi/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "orchestrator_result" in data


def test_omi_webhook_empty_payload_ignored(client):
    """POST /api/omi/webhook with no transcript returns ignored status."""
    payload = {"event": "ping", "data": {}}
    response = client.post("/api/omi/webhook", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


# ─── OmiVoiceListener Unit Tests ─────────────────────────────────────────────

def test_omi_listener_process_transcript_unit(in_memory_orchestrator):
    """Unit test: OmiVoiceListener.process_voice_transcript returns proper structure."""
    listener = OmiVoiceListener(orchestrator=in_memory_orchestrator)
    result = listener.process_voice_transcript(
        transcript="Schedule a review call at 3pm",
        speaker="ravel",
        session_id="unit-test-session",
    )
    assert result["status"] == "success"
    assert result["source"] == "omi_voice"
    assert result["transcript"] == "Schedule a review call at 3pm"
    assert result["speech_response"] is not None
    assert "orchestrator_result" in result


def test_omi_listener_handle_webhook_event_unit(in_memory_orchestrator):
    """Unit test: OmiVoiceListener.handle_webhook_event extracts transcript from payload."""
    listener = OmiVoiceListener(orchestrator=in_memory_orchestrator)
    payload = {
        "event": "memory_created",
        "data": {
            "text": "Remember to submit the project by end of day",
            "speaker": "ravel",
        },
    }
    result = listener.handle_webhook_event(payload)
    assert result["status"] == "success"
    assert "orchestrator_result" in result
