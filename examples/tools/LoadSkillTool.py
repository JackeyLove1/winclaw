from typing import Any

from examples.base import Tool, ToolResult
from examples.skill_loader import SkillLoader


class LoadSkillTool(Tool):
    def __init__(self, loader: SkillLoader) -> None:
        self._loader = loader

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        return (
            "Load the full body of a bundled skill by name (from examples/skills/*/SKILL.md). "
            "Call this before following specialized workflows you do not already know."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill id as listed in the system prompt (frontmatter name or folder)",
                },
            },
            "required": ["name"],
        }

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        content = self._loader.get_content(str(kwargs["name"]).strip())
        return ToolResult(success=True, content=content)
