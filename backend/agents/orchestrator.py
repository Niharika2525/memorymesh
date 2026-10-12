"""Orchestrator Agent module.

The central coordinator that receives tasks (from voice listener or API),
coordinates multi-agent collaboration across Memory, Planner, Research, Action,
and Verifier agents, and writes new contextual knowledge back into Qdrant.
"""

import uuid
from typing import Any, Dict, Optional

from backend.agents.action import ActionAgent
from backend.agents.memory import MemoryAgent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.verifier import VerifierAgent
from backend.services.llm import LLMService, get_llm_service


class OrchestratorAgent:
    """Coordinates and orchestrates tasks across specialized agents."""

    def __init__(
        self,
        memory_agent: Optional[MemoryAgent] = None,
        planner_agent: Optional[PlannerAgent] = None,
        research_agent: Optional[ResearchAgent] = None,
        action_agent: Optional[ActionAgent] = None,
        verifier_agent: Optional[VerifierAgent] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.memory = memory_agent or MemoryAgent()
        self.planner = planner_agent or PlannerAgent()
        self.research = research_agent or ResearchAgent()
        self.action = action_agent or ActionAgent()
        self.verifier = verifier_agent or VerifierAgent()
        self.llm = llm_service or get_llm_service()

    def run_workflow(
        self,
        task: str,
        speaker: Optional[str] = "user",
        source: Optional[str] = "voice",
        auto_persist_memory: bool = True,
    ) -> Dict[str, Any]:
        """Execute the end-to-end Voice -> Memory -> Agent execution loop."""
        execution_id = str(uuid.uuid4())

        # Step 1: Query persistent vector memory in Qdrant for past context
        recalled_context = self.memory.recall_context_for_prompt(query_text=task, limit=3)
        raw_memories = self.memory.recall(query_text=task, limit=3)

        # Step 2: Planner breaks down user task
        plan = self.planner.create_plan(task=task, memory_context=recalled_context)

        # Step 3: Research Agent synthesizes knowledge
        research_res = self.research.analyze(query=task, context=recalled_context)

        # Step 4: Action Agent executes tasks
        # Detect if math or report or generic
        action_params = {"task": task, "speaker": speaker, "context": recalled_context}
        action_tool = "generic"
        if any(c in task for c in ["+", "-", "*", "/"]) and any(c.isdigit() for c in task):
            action_tool = "calculate"
            action_params["expression"] = "".join(c for c in task if c in "0123456789+-*/() ")

        action_res = self.action.execute(action_name=action_tool, parameters=action_params)

        # Step 5: Verifier Agent checks output
        execution_steps_results = [
            {"step": "research", "status": research_res.get("status"), "details": research_res},
            {"step": "action", "status": action_res.get("status"), "details": action_res},
        ]
        verification = self.verifier.verify(task=task, results=execution_steps_results)

        # Step 6: Final response synthesis
        synthesis_prompt = (
            f"User Task: {task}\n"
            f"Prior Context: {recalled_context}\n"
            f"Research: {research_res.get('findings')}\n"
            f"Action Result: {action_res.get('result')}\n\n"
            f"Synthesize a clear, helpful response for the user."
        )
        
        from lyzr_automata import Agent, Task
        from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
        from lyzr_automata.tasks.task_literals import InputType, OutputType
        from backend.services.llm import LyzrMemoryMeshModel

        # Phase 3 Lyzr Integration: Define the orchestrator as a Lyzr Agent
        orchestrator_agent = Agent(
            role="Orchestrator",
            prompt_persona="You are MemoryMesh, a voice-first autonomous multi-agent assistant."
        )

        lyzr_model = LyzrMemoryMeshModel(self.llm)

        # Define the synthesis as a Lyzr Task
        synthesis_task = Task(
            name="Synthesize Final Response",
            model=lyzr_model,
            agent=orchestrator_agent,
            instructions=synthesis_prompt,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
        )

        # Execute via Lyzr Pipeline
        pipeline = LinearSyncPipeline(
            name="MemoryMesh Orchestration",
            completion_message="Orchestration completed",
            tasks=[synthesis_task]
        )
        
        pipeline_output = pipeline.run()
        final_response = pipeline_output[0]['task_output']

        # Step 7: Write new contextual knowledge back into Qdrant vector memory
        persisted_memory = None
        if auto_persist_memory:
            persisted_memory = self.memory.remember(
                content=f"Task: {task} | Outcome: {final_response.strip()}",
                metadata={
                    "speaker": speaker,
                    "source": source,
                    "execution_id": execution_id,
                    "verified": verification["verified"],
                },
            )

        provider_used = self.llm.last_provider_used
        lyzr_error = self.llm.last_lyzr_error

        return {
            "execution_id": execution_id,
            "task": task,
            "speaker": speaker,
            "source": source,
            "llm_provider": provider_used,
            "lyzr_status": {
                "used": provider_used == "lyzr",
                "provider": provider_used,
                "error": lyzr_error if provider_used != "lyzr" else None,
            },
            "recalled_memories_count": len(raw_memories),
            "recalled_memories": raw_memories,
            "plan": plan,
            "research": research_res,
            "action": action_res,
            "verification": verification,
            "persisted_memory": persisted_memory,
            "final_response": final_response.strip(),
        }


# Global singleton
_orchestrator_instance: Optional[OrchestratorAgent] = None


def get_orchestrator_agent() -> OrchestratorAgent:
    """Dependency helper returning active OrchestratorAgent instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance
