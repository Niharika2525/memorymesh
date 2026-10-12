"""Memory Agent module.

Interfaces with Qdrant vector memory service to recall relevant past interactions,
user preferences, and context, and persist valuable new insights for agent workflows.
"""

from typing import Any, Dict, List, Optional

from backend.memory.qdrant_client import QdrantMemoryService, get_qdrant_service


class MemoryAgent:
    """Coordinates memory persistence and semantic recall for multi-agent workflows."""

    def __init__(self, memory_service: Optional[QdrantMemoryService] = None):
        """Initialize MemoryAgent with the Qdrant memory service."""
        self.memory_service = memory_service or get_qdrant_service()

    def check_health(self) -> Dict[str, Any]:
        """Verify the health and readiness of the memory storage."""
        return self.memory_service.check_connection()

    def remember(
        self,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a new memory embedding along with contextual metadata."""
        return self.memory_service.store_memory(
            content=content,
            vector=vector,
            metadata=metadata,
            memory_id=memory_id,
        )

    def recall(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past memories using semantic similarity."""
        return self.memory_service.search_memories(
            query_vector=query_vector,
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
            filter_criteria=filter_criteria,
        )

    def recall_context_for_prompt(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> str:
        """Recall relevant memories and format them for injection into agent prompts."""
        memories = self.recall(
            query_vector=query_vector,
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
        )
        return self.memory_service.format_memories_for_agent(memories)
