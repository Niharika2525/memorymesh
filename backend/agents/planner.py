"""Planner Agent module.

Deconstructs high-level user tasks into sequential and parallel execution steps,
assigning specific roles and tool requirements across the agent mesh.
"""

from typing import Any, Dict, List, Optional
from backend.services.llm import LLMService, get_llm_service


class PlannerAgent:
    """Generates execution plans and step-by-step strategies for incoming tasks."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def create_plan(
        self,
        task: str,
        memory_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a structured execution plan for the requested task."""
        context_block = f"Prior Knowledge & Context:\n{memory_context}\n" if memory_context else ""
        prompt = (
            f"{context_block}"
            f"User Objective: {task}\n\n"
            f"Decompose this objective into actionable agent execution steps."
        )
        system_prompt = (
            "You are the Planner Agent in MemoryMesh. Break down the user objective into "
            "clear steps assigned to Research, Action, or Verifier agents."
        )

        from lyzr_automata import Agent, Task
        from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
        from lyzr_automata.tasks.task_literals import InputType, OutputType
        from backend.services.llm import LyzrMemoryMeshModel

        lyzr_model = LyzrMemoryMeshModel(self.llm_service)
        planner_lyzr_agent = Agent(role="Planner Agent", prompt_persona=system_prompt)
        planner_task = Task(
            name="Create Execution Plan",
            model=lyzr_model,
            agent=planner_lyzr_agent,
            instructions=prompt,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
        )
        pipeline = LinearSyncPipeline(name="Planner Pipeline", tasks=[planner_task])
        plan_text = pipeline.run()[0]['task_output']

        # Build structured step list
        steps: List[Dict[str, Any]] = [
            {
                "step": 1,
                "agent": "research",
                "action": "analyze_context",
                "description": f"Analyze objective '{task}' against recalled memory context.",
            },
            {
                "step": 2,
                "agent": "action",
                "action": "execute_task",
                "description": f"Execute required tools and syntheses for objective '{task}'.",
            },
            {
                "step": 3,
                "agent": "verifier",
                "action": "validate_outcome",
                "description": "Verify execution results against the initial user objective.",
            },
        ]

        return {
            "task": task,
            "has_memory_context": bool(memory_context),
            "plan_summary": plan_text.strip(),
            "steps": steps,
        }
