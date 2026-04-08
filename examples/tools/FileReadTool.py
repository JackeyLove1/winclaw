import asyncio
from pathlib import Path
from typing import Any

from examples.base import Tool, ToolResult
from examples.tools.workspace_fs import read_workspace_file, safe_workspace_path


class FileReadTool(Tool):
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents from the workspace. Each line is "
            "'LINE_NUMBER|LINE_CONTENT' (1-indexed). For large files use offset and limit "
            "to read in chunks. Parallel calls can read different files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to the example workspace)",
                },
                "offset": {
                    "type": "integer",
                    "description": "First line to return (1-indexed). Omit to start at line 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of lines to return. Omit to read through end of file",
                },
            },
            "required": ["path"],
        }

    def _read_sync(self, path: str, offset: int | None, limit: int | None) -> str:
        try:
            text, _ = read_workspace_file(safe_workspace_path(self._workdir, path))
            lines = text.splitlines()
            total = len(lines)
            start_line = 1 if offset is None else int(offset)
            if start_line < 1:
                return "Error: offset must be >= 1"
            if total == 0:
                return "" if start_line == 1 else "Error: offset is beyond end of file"
            idx0 = start_line - 1
            if idx0 >= total:
                return "Error: offset is beyond end of file"
            if limit is None:
                end_idx = total
            else:
                lim = int(limit)
                if lim < 0:
                    return "Error: limit must be >= 0"
                end_idx = min(idx0 + lim, total)
            chunk = lines[idx0:end_idx]
            numbered = [f"{start_line + i}|{line}" for i, line in enumerate(chunk)]
            more_after = total - end_idx
            if limit is not None and more_after > 0:
                numbered.append(f"... ({more_after} more lines in file)")
            return "\n".join(numbered)[:50000]
        except Exception as e:
            return f"Error: {e}"

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        content = await asyncio.to_thread(
            self._read_sync,
            kwargs["path"],
            kwargs.get("offset"),
            kwargs.get("limit"),
        )
        return ToolResult(success=True, content=content)
