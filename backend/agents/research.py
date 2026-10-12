"""Research / Reasoning Agent module.

Conducts in-depth analytical reasoning, context synthesis, and information gathering
required to solve complex user objectives.
"""

from typing import Any, Dict, Optional
from backend.services.llm import LLMService, get_llm_service


class ResearchAgent:
    """Performs deep reasoning and analytical synthesis."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or get_llm_service()

    def analyze(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform analytical reasoning over the query and contextual memories."""
        prompt = (
            f"Context:\n{context or 'No prior context provided.'}\n\n"
            f"Task / Question:\n{query}\n\n"
            f"Provide an analytical synthesis and identify key facts or decisions."
        )
        system_prompt = (
            "You are the Research & Reasoning Agent in MemoryMesh. Synthesize information, "
            "highlight relevant past memory facts, and deduce next steps."
        )

        from lyzr_automata import Agent, Task
        from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
        from lyzr_automata.tasks.task_literals import InputType, OutputType
        from backend.services.llm import LyzrMemoryMeshModel

        lyzr_model = LyzrMemoryMeshModel(self.llm_service)
        research_lyzr_agent = Agent(role="Research Agent", prompt_persona=system_prompt)
        research_task = Task(
            name="Analyze Context",
            model=lyzr_model,
            agent=research_lyzr_agent,
            instructions=prompt,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
        )
        pipeline = LinearSyncPipeline(name="Research Pipeline", tasks=[research_task])
        analysis_text = pipeline.run()[0]['task_output']

        return {
            "query": query,
            "status": "completed",
            "findings": analysis_text.strip(),
            "context_utilized": bool(context),
        }
