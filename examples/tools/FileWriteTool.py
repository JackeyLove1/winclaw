import asyncio
from pathlib import Path
from typing import Any

from examples.base import Tool, ToolResult
from examples.tools.workspace_fs import safe_workspace_path


class FileWriteTool(Tool):
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

    def _write_sync(self, path: str, content: str) -> str:
        try:
            fp = safe_workspace_path(self._workdir, path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        text = await asyncio.to_thread(
            self._write_sync,
            kwargs["path"],
            kwargs["content"],
        )
        return ToolResult(success=True, content=text)
