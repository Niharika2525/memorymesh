"""FastAPI router for the multi-agent orchestration workflow.

Exposes the Orchestrator Agent run_workflow loop via REST,
allowing direct task submission without going through voice input.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent

router = APIRouter()


class TaskRequest(BaseModel):
    """Request payload for submitting a task to the agent mesh."""

    task: str = Field(..., description="Natural language task or question for the agent mesh")
    speaker: str = Field(default="user", description="Speaker or user identity")
    source: str = Field(default="api", description="Originating source: api, voice, webhook, etc.")
    auto_persist_memory: bool = Field(default=True, description="Whether to persist the outcome in Qdrant")


@router.post("/run", tags=["Agent Mesh"])
async def run_workflow(
    request: TaskRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator_agent),
) -> Dict[str, Any]:
    """Submit a task to the autonomous multi-agent mesh and return the full execution result."""
    try:
        result = orchestrator.run_workflow(
            task=request.task,
            speaker=request.speaker,
            source=request.source,
            auto_persist_memory=request.auto_persist_memory,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow execution failed: {str(e)}",
        )


@router.get("/status", tags=["Agent Mesh"])
async def agent_status(
    orchestrator: OrchestratorAgent = Depends(get_orchestrator_agent),
) -> Dict[str, Any]:
    """Check agent mesh readiness and Qdrant memory connection status."""
    memory_status = orchestrator.memory.check_health()
    lyzr_status = orchestrator.llm.check_lyzr_status()
    return {
        "status": "ready",
        "agents": ["orchestrator", "planner", "memory", "research", "action", "verifier"],
        "memory_connected": memory_status.get("connected", False),
        "memory_collection": memory_status.get("collection"),
        "memory_collection_exists": memory_status.get("collection_exists", False),
        "lyzr": lyzr_status,
    }
