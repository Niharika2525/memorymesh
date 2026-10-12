"""Verifier Agent module.

Validates execution results against the original user prompt and objectives,
ensuring correctness, alignment, and completeness before returning results.
"""

from typing import Any, Dict, List, Optional
from backend.services.llm import LLMService, get_llm_service


class VerifierAgent:
    """Verifies that execution outcomes meet quality and objective standards."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def verify(
        self,
        task: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate execution results against the original task."""
        results_summary = "\n".join(f"- {r.get('step', idx)}: {r.get('status')}" for idx, r in enumerate(results, 1))
        prompt = (
            f"User Objective:\n{task}\n\n"
            f"Execution Steps Results:\n{results_summary}\n\n"
            f"Validate if the task has been adequately and accurately addressed."
        )
        system_prompt = (
            "You are the Verifier Agent in MemoryMesh. Evaluate completeness, consistency, "
            "and alignment with the initial goal."
        )

        from lyzr_automata import Agent, Task
        from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
        from lyzr_automata.tasks.task_literals import InputType, OutputType
        from backend.services.llm import LyzrMemoryMeshModel

        lyzr_model = LyzrMemoryMeshModel(self.llm_service)
        verifier_lyzr_agent = Agent(role="Verifier Agent", prompt_persona=system_prompt)
        verifier_task = Task(
            name="Verify Results",
            model=lyzr_model,
            agent=verifier_lyzr_agent,
            instructions=prompt,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
        )
        pipeline = LinearSyncPipeline(name="Verifier Pipeline", tasks=[verifier_task])
        verification_text = pipeline.run()[0]['task_output']
        
        all_passed = all(r.get("status") in ("success", "completed") for r in results)

        return {
            "verified": all_passed,
            "confidence_score": 0.95 if all_passed else 0.50,
            "verification_notes": verification_text.strip(),
            "steps_evaluated": len(results),
        }
