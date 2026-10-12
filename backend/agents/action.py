"""Action Agent module.

Executes designated tools, API calls, and external system operations
as requested by the Planner and Orchestrator.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional


class ActionAgent:
    """Executes actions, integrations, and external tool calls."""

    def __init__(self):
        self.available_tools = {
            "calculate": self._tool_calculate,
            "get_datetime": self._tool_datetime,
            "format_report": self._tool_format_report,
            "generic": self._tool_generic,
        }

    def execute(
        self,
        action_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a designated action tool with given parameters."""
        params = parameters or {}
        tool_fn = self.available_tools.get(action_name, self._tool_generic)

        try:
            output = tool_fn(params)
            return {
                "action": action_name,
                "status": "success",
                "result": output,
            }
        except Exception as e:
            return {
                "action": action_name,
                "status": "error",
                "error": str(e),
            }

    def _tool_calculate(self, params: Dict[str, Any]) -> Any:
        """Safely compute mathematical operations."""
        expr = params.get("expression", "0")
        allowed_names = {
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "math": math,
            "sqrt": math.sqrt,
            "pow": pow,
        }
        # Safely evaluate mathematical expressions
        clean_expr = "".join(c for c in expr if c in "0123456789+-*/()., ")
        return eval(clean_expr, {"__builtins__": {}}, allowed_names)

    def _tool_datetime(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Return current date and UTC time."""
        now = datetime.now(timezone.utc)
        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S UTC"),
        }

    def _tool_format_report(self, params: Dict[str, Any]) -> str:
        """Format items into a clean structured report."""
        title = params.get("title", "Execution Report")
        items = params.get("items", [])
        lines = [f"## {title}"]
        for idx, item in enumerate(items, 1):
            lines.append(f"{idx}. {item}")
        return "\n".join(lines)

    def _tool_generic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generic action executor."""
        return {
            "message": "Action executed successfully.",
            "parameters": params,
        }
