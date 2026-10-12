"""Pydantic models for MemoryMesh memory service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryStoreRequest(BaseModel):
    """Request model for storing a memory embedding."""

    content: str = Field(..., description="Text content of the memory")
    vector: List[float] = Field(..., description="Vector embedding for the memory content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata tags, source, or session info")
    id: Optional[str] = Field(default=None, description="Optional custom ID for the memory point")


class MemorySearchRequest(BaseModel):
    """Request model for semantic similarity search."""

    vector: List[float] = Field(..., description="Query vector embedding")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of results to return")
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity score")
    filter_criteria: Optional[Dict[str, Any]] = Field(default=None, description="Key-value filters on metadata")


class MemoryItemResponse(BaseModel):
    """Response model for a single retrieved memory."""

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    score: Optional[float] = None


class MemorySearchResponse(BaseModel):
    """Response model for semantic search results."""

    count: int
    memories: List[MemoryItemResponse]
    context_text: str = Field(description="Formatted context ready for agent prompt injection")
