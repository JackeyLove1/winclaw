from collections.abc import Awaitable, Callable
from typing import Any

from examples.base import Tool, ToolResult


class TaskTool(Tool):
    def __init__(self, run_subagent: Callable[[str], Awaitable[str]]) -> None:
        self._run_subagent = run_subagent

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent with fresh context. It shares the filesystem but not "
            "conversation history, and returns only a summary."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short description of the delegated task",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        summary = await self._run_subagent(kwargs["prompt"])
        return ToolResult(success=True, content=summary)
