"""FastAPI router for Qdrant memory operations."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from backend.memory.models import (
    MemoryItemResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStoreRequest,
)
from backend.memory.qdrant_client import QdrantMemoryService, get_qdrant_service

router = APIRouter()


@router.get("/health", tags=["Memory"])
async def memory_health(
    service: QdrantMemoryService = Depends(get_qdrant_service),
) -> Dict[str, Any]:
    """Check Qdrant vector database connection status."""
    return service.check_connection()


@router.post("/store", response_model=Dict[str, Any], tags=["Memory"])
async def store_memory(
    request: MemoryStoreRequest,
    service: QdrantMemoryService = Depends(get_qdrant_service),
) -> Dict[str, Any]:
    """Store a memory embedding with associated metadata."""
    try:
        result = service.store_memory(
            content=request.content,
            vector=request.vector,
            metadata=request.metadata,
            memory_id=request.id,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}",
        )


@router.post("/search", response_model=MemorySearchResponse, tags=["Memory"])
async def search_memories(
    request: MemorySearchRequest,
    service: QdrantMemoryService = Depends(get_qdrant_service),
) -> MemorySearchResponse:
    """Semantic similarity search returning matching memories and agent-ready context."""
    try:
        memories = service.search_memories(
            query_vector=request.vector,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filter_criteria=request.filter_criteria,
        )
        context_text = service.format_memories_for_agent(memories)
        return MemorySearchResponse(
            count=len(memories),
            memories=[MemoryItemResponse(**m) for m in memories],
            context_text=context_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}",
        )


@router.get("/{memory_id}", response_model=MemoryItemResponse, tags=["Memory"])
async def get_memory(
    memory_id: str,
    service: QdrantMemoryService = Depends(get_qdrant_service),
) -> MemoryItemResponse:
    """Retrieve a single memory by ID."""
    memory = service.get_memory(memory_id=memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with ID '{memory_id}' not found.",
        )
    return MemoryItemResponse(**memory)


@router.delete("/{memory_id}", tags=["Memory"])
async def delete_memory(
    memory_id: str,
    service: QdrantMemoryService = Depends(get_qdrant_service),
) -> Dict[str, Any]:
    """Delete a memory by ID."""
    success = service.delete_memory(memory_id=memory_id)
    return {"id": memory_id, "deleted": success}
