"""Focused tests for QdrantMemoryService and MemoryAgent."""

import pytest
from unittest.mock import MagicMock
from qdrant_client import QdrantClient

from backend.memory.qdrant_client import QdrantMemoryService, _mask_url
from backend.agents.memory import MemoryAgent


@pytest.fixture
def memory_service():
    """Create an isolated in-memory Qdrant service for testing."""
    client = QdrantClient(":memory:")
    service = QdrantMemoryService(
        url=":memory:",
        collection_name="test_memories",
        vector_size=4,
        distance="Cosine",
        client=client,
    )
    return service


def test_mask_url_hides_sensitive_info():
    """Ensure URL masking never exposes passwords or tokens."""
    assert _mask_url(":memory:") == ":memory:"
    assert _mask_url("http://localhost:6333") == "http://localhost:6333"
    masked = _mask_url("https://user:supersecretpass@cluster.qdrant.tech:6333")
    assert "supersecretpass" not in masked
    assert "***" in masked


def test_qdrant_service_check_connection_success(memory_service):
    """Verify check_connection returns healthy status for connected cluster."""
    status = memory_service.check_connection()
    assert status["connected"] is True
    assert status["status"] == "connected"
    assert status["collection"] == "test_memories"
    assert status["error"] is None


def test_qdrant_service_check_connection_failure():
    """Verify check_connection handles connection error gracefully without leaking secrets."""
    mock_client = MagicMock()
    secret_key = "super_confidential_api_key_123"
    mock_client.get_collections.side_effect = Exception(f"Failed to authenticate with key {secret_key}")

    service = QdrantMemoryService(
        url="http://fake-cluster:6333",
        api_key=secret_key,
        client=mock_client,
    )
    status = service.check_connection()
    assert status["connected"] is False
    assert status["status"] == "disconnected"
    assert secret_key not in status["error"]
    assert "***" in status["error"]


def test_ensure_collection(memory_service):
    """Verify ensure_collection creates collection idempotently."""
    assert memory_service.client.collection_exists("test_memories") is False
    created = memory_service.ensure_collection()
    assert created is True
    assert memory_service.client.collection_exists("test_memories") is True

    # Re-running must be safe and idempotent
    assert memory_service.ensure_collection() is True


def test_store_and_get_memory(memory_service):
    """Verify storing a memory embedding with metadata and retrieving it."""
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {
        "source": "omi_voice",
        "speaker": "ravel",
        "tags": ["meeting", "project"],
    }
    result = memory_service.store_memory(
        content="Discussed Qdrant integration for MemoryMesh hackathon",
        vector=vector,
        metadata=metadata,
        memory_id="mem-12345",
    )

    assert result["id"] == "mem-12345"
    assert result["status"] == "stored"
    assert result["content"] == "Discussed Qdrant integration for MemoryMesh hackathon"
    assert result["metadata"]["source"] == "omi_voice"
    assert "created_at" in result

    # Retrieve memory
    retrieved = memory_service.get_memory("mem-12345")
    assert retrieved is not None
    assert retrieved["id"] == "mem-12345"
    assert retrieved["content"] == "Discussed Qdrant integration for MemoryMesh hackathon"
    assert retrieved["metadata"]["speaker"] == "ravel"


def test_store_memories_batch(memory_service):
    """Verify batch storing of multiple memories."""
    memories = [
        {
            "id": "batch-1",
            "content": "First memory item",
            "vector": [0.1, 0.1, 0.1, 0.1],
            "metadata": {"source": "omi"},
        },
        {
            "id": "batch-2",
            "content": "Second memory item",
            "vector": [0.2, 0.2, 0.2, 0.2],
            "metadata": {"source": "notes"},
        },
    ]
    ids = memory_service.store_memories_batch(memories)
    assert ids == ["batch-1", "batch-2"]
    assert memory_service.count_memories() == 2


def test_search_memories_semantic_ranking(memory_service):
    """Verify semantic similarity search returns ranked results."""
    # Store 3 items with distinct vectors
    memory_service.store_memory(
        content="Target meeting notes about AI architecture",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"category": "tech"},
    )
    memory_service.store_memory(
        content="Irrelevant grocery shopping list",
        vector=[0.0, 1.0, 0.0, 0.0],
        metadata={"category": "personal"},
    )
    memory_service.store_memory(
        content="Semi-relevant system engineering discussion",
        vector=[0.8, 0.2, 0.0, 0.0],
        metadata={"category": "tech"},
    )

    # Search with a vector closest to [1.0, 0.0, 0.0, 0.0]
    results = memory_service.search_memories(
        query_vector=[0.95, 0.05, 0.0, 0.0],
        limit=2,
    )

    assert len(results) == 2
    assert "Target meeting notes" in results[0]["content"]
    assert results[0]["score"] > results[1]["score"]


def test_search_memories_with_filter(memory_service):
    """Verify filtering search results by metadata fields."""
    memory_service.store_memory(
        content="Tech item 1",
        vector=[0.5, 0.5, 0.0, 0.0],
        metadata={"source": "omi_voice"},
    )
    memory_service.store_memory(
        content="Tech item 2",
        vector=[0.5, 0.5, 0.0, 0.0],
        metadata={"source": "manual_entry"},
    )

    results = memory_service.search_memories(
        query_vector=[0.5, 0.5, 0.0, 0.0],
        limit=5,
        filter_criteria={"source": "omi_voice"},
    )

    assert len(results) == 1
    assert results[0]["content"] == "Tech item 1"


def test_delete_memory(memory_service):
    """Verify memory deletion."""
    memory_service.store_memory(
        content="Temporary note to delete",
        vector=[0.1, 0.2, 0.3, 0.4],
        memory_id="temp-mem",
    )
    assert memory_service.count_memories() == 1

    deleted = memory_service.delete_memory("temp-mem")
    assert deleted is True
    assert memory_service.get_memory("temp-mem") is None
    assert memory_service.count_memories() == 0


def test_format_memories_for_agent():
    """Verify formatting memory items into context string for agent workflow."""
    memories = [
        {
            "id": "1",
            "score": 0.945,
            "content": "User prefers concise python answers.",
            "metadata": {"source": "omi_voice"},
            "created_at": "2026-10-01T12:00:00Z",
        },
        {
            "id": "2",
            "score": 0.812,
            "content": "User is building MemoryMesh hackathon project.",
            "metadata": {"source": "lyzr_agent"},
            "created_at": "2026-10-01T13:00:00Z",
        },
    ]

    formatted = QdrantMemoryService.format_memories_for_agent(memories)
    assert "### Relevant Retrieved Context & Memories:" in formatted
    assert "[relevance:" in formatted
    assert "User prefers concise python answers." in formatted
    assert "User is building MemoryMesh hackathon project." in formatted

    empty_formatted = QdrantMemoryService.format_memories_for_agent([])
    assert "No previous relevant memories found." in empty_formatted


def test_memory_agent_integration(memory_service):
    """Verify MemoryAgent properly delegates to QdrantMemoryService."""
    agent = MemoryAgent(memory_service=memory_service)

    # Health
    health = agent.check_health()
    assert health["connected"] is True

    # Remember
    stored = agent.remember(
        content="Agent remembered user preference for morning briefings.",
        vector=[0.2, 0.4, 0.6, 0.8],
        metadata={"priority": "high"},
    )
    assert stored["status"] == "stored"

    # Recall
    recalled = agent.recall(query_vector=[0.2, 0.4, 0.6, 0.8], limit=1)
    assert len(recalled) == 1
    assert "morning briefings" in recalled[0]["content"]

    # Recall prompt context
    prompt_context = agent.recall_context_for_prompt(query_vector=[0.2, 0.4, 0.6, 0.8], limit=1)
    assert "morning briefings" in prompt_context
    assert "Relevant Retrieved Context" in prompt_context
