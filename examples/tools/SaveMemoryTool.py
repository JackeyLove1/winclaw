from typing import Any

from examples.base import Tool, ToolResult
from examples.memory.memory_manager import MEMORY_TYPES, MemoryManager


class SaveMemoryTool(Tool):
    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save a persistent memory that survives across sessions (under .memory/). "
            "Use only for cross-session facts worth recalling, not for repo structure or secrets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short identifier (e.g. prefer_tabs, db_schema)",
                },
                "description": {
                    "type": "string",
                    "description": "One-line summary of what this memory captures",
                },
                "type": {
                    "type": "string",
                    "enum": list(MEMORY_TYPES),
                    "description": (
                        "user=preferences, feedback=corrections, "
                        "project=non-obvious conventions or decision reasons, "
                        "reference=external resource pointers"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full memory content (multi-line OK)",
                },
            },
            "required": ["name", "description", "type", "content"],
        }

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        msg = self._memory.save_memory(
            str(kwargs["name"]).strip(),
            str(kwargs["description"]).strip(),
            str(kwargs["type"]).strip(),
            str(kwargs["content"]),
        )
        if msg.startswith("Error:"):
            return ToolResult(success=False, content="", error=msg)
        return ToolResult(success=True, content=msg)
