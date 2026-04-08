from collections.abc import Awaitable, Callable
from typing import Any

from examples.base import Tool, ToolResult


class ForkTool(Tool):
    def __init__(
        self,
        run_fork_agent: Callable[[list[Any], str], Awaitable[str]],
        get_parent_messages: Callable[[], list[Any] | None],
    ) -> None:
        self._run_fork_agent = run_fork_agent
        self._get_parent_messages = get_parent_messages

    @property
    def name(self) -> str:
        return "fork"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent that inherits the parent's chat history (through the last completed turn), "
            "then follows your prompt. Use when the work needs prior decisions, paths, or errors from this "
            "conversation. Same tools and workspace as the parent; no task/fork inside the fork. Returns a "
            "summary only."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short label for this fork (optional)",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        parent = self._get_parent_messages()
        if parent is None:
            return ToolResult(
                success=False,
                content="",
                error="fork could not read parent messages (internal)",
            )
        summary = await self._run_fork_agent(parent, kwargs["prompt"])
        return ToolResult(success=True, content=summary)
