"""MemoryMesh autonomous multi-agent mesh package.

Exports the 6 core agent roles:
1. OrchestratorAgent - Central coordinator receiving voice instructions and directing execution.
2. PlannerAgent - Deconstructs tasks into structured execution plans.
3. MemoryAgent - Interfaces with Qdrant vector memory for semantic search and persistence.
4. ResearchAgent - Performs analytical synthesis and reasoning.
5. ActionAgent - Executes tools, calculations, and integrations.
6. VerifierAgent - Validates execution quality against the user objective.
"""

from backend.agents.action import ActionAgent
from backend.agents.memory import MemoryAgent
from backend.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.verifier import VerifierAgent

__all__ = [
    "OrchestratorAgent",
    "get_orchestrator_agent",
    "PlannerAgent",
    "MemoryAgent",
    "ResearchAgent",
    "ActionAgent",
    "VerifierAgent",
]
