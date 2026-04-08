import asyncio
from pathlib import Path
from typing import Any

from examples.base import Tool, ToolResult
from examples.tools.workspace_fs import read_workspace_file, safe_workspace_path


class FileEditTool(Tool):
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace exact text in file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    def _edit_sync(self, path: str, old_text: str, new_text: str) -> str:
        try:
            fp = safe_workspace_path(self._workdir, path)
            content, enc = read_workspace_file(fp)
            if old_text not in content:
                return f"Error: Text not found in {path}"
            fp.write_text(content.replace(old_text, new_text, 1), encoding=enc)
            return f"Edited {path}"
        except Exception as e:
            return f"Error: {e}"

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        text = await asyncio.to_thread(
            self._edit_sync,
            kwargs["path"],
            kwargs["old_text"],
            kwargs["new_text"],
        )
        return ToolResult(success=True, content=text)
