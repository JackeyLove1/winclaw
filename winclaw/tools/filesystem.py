"""
File system tools: read, write, edit.
TODO: disable filesystem tools by default, use powershell for more flexibility.
TODO: support glob and rg
"""

import asyncio
import difflib
import locale
import os
import sys
from pathlib import Path
from typing import Any, Optional

from winclaw.tools.base import Tool
from winclaw.utils.helpers import get_bin_path


def _resolve_path(
    path: str, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None
) -> Path:
    """Resolve path against workspace (if relative) and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class ReadFileTool(Tool):
    """Tool to read file contents."""

    _MAX_CHARS = 128_000  # ~128 KB — prevents OOM from reading huge files into LLM context

    def __init__(self, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The file path to read"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            size = file_path.stat().st_size
            if size > self._MAX_CHARS * 4:  # rough upper bound (UTF-8 chars ≤ 4 bytes)
                return (
                    f"Error: File too large ({size:,} bytes). "
                    f"Use exec tool with head/tail/grep to read portions."
                )

            content = file_path.read_text(encoding="utf-8")
            if len(content) > self._MAX_CHARS:
                return (
                    content[: self._MAX_CHARS]
                    + f"\n\n... (truncated — file is {len(content):,} chars, limit {self._MAX_CHARS:,})"
                )
            return content
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    def __init__(self, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text."""

    def __init__(self, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, path: str, old_text: str, new_text: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"

            content = file_path.read_text(encoding="utf-8")

            if old_text not in content:
                return self._not_found_message(old_text, content, path)

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return f"Warning: old_text appears {count} times. Please provide more context to make it unique."

            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")

            return f"Successfully edited {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    @staticmethod
    def _not_found_message(old_text: str, content: str, path: str) -> str:
        """Build a helpful error when old_text is not found."""
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio > 0.5:
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    lines[best_start : best_start + window],
                    fromfile="old_text (provided)",
                    tofile=f"{path} (actual, line {best_start + 1})",
                    lineterm="",
                )
            )
            return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return (
            f"Error: old_text not found in {path}. No similar text found. Verify the file content."
        )


class ListDirTool(Tool):
    """Tool to list directory contents."""

    def __init__(self, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The directory path to list"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            dir_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class GrepTool(Tool):
    """Tool to grep file contents."""

    _MAX_OUTPUT_CHARS = 10_000
    _TIMEOUT_SECONDS = 30

    def __init__(
        self,
        workspace: Optional[Path] = None,
        allowed_dir: Optional[Path] = None,
        bin_path: Optional[Path] = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._bin_path = bin_path or get_bin_path()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with bundled ripgrep."
            + "High-performance text search tool, suitable for quickly locating symbols, config keys, and error messages in large repositories"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The ripgrep pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Optional file or directory path to search in",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional ripgrep glob filter such as '*.py'",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Choose matching lines, matched files, or per-file counts",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Whether the search should ignore case",
                },
                "context_lines": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "How many context lines to include around each match",
                },
                "head_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of output lines to return",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        output_mode: str = "content",
        ignore_case: bool = False,
        context_lines: int = 0,
        head_limit: int = 50,
        **kwargs: Any,
    ) -> str:
        try:
            search_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not search_path.exists():
                return f"Error: Search path not found: {path}"

            rg_path = self._bin_path / "rg.exe"
            if not rg_path.exists():
                return f"Error: rg executable not found: {rg_path}"

            command = self._build_command(
                rg_path=rg_path,
                pattern=pattern,
                path=search_path,
                glob=glob,
                output_mode=output_mode,
                ignore_case=ignore_case,
                context_lines=context_lines,
            )

            cwd = self._workspace or (search_path.parent if search_path.is_file() else search_path)

            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=os.environ.copy(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self._TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return f"Error: grep timed out after {self._TIMEOUT_SECONDS} seconds"

            decode_encoding = self._best_effort_output_encoding()
            stdout_text = self._decode_output(stdout, decode_encoding).strip()
            stderr_text = self._decode_output(stderr, decode_encoding).strip()

            if process.returncode == 1 and not stderr_text:
                return f"No matches found for pattern: {pattern}"
            if process.returncode not in (0, 1):
                detail = (
                    stderr_text or stdout_text or f"ripgrep exited with code {process.returncode}"
                )
                return f"Error running grep: {detail}"

            result = stdout_text or "(no output)"
            result = self._truncate_lines(result, head_limit)
            if len(result) > self._MAX_OUTPUT_CHARS:
                result = (
                    result[: self._MAX_OUTPUT_CHARS]
                    + f"\n... (truncated, {len(result) - self._MAX_OUTPUT_CHARS} more chars)"
                )

            if stderr_text:
                result += f"\n\nSTDERR:\n{stderr_text}"

            return result
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error running grep: {str(e)}"

    @classmethod
    def _build_command(
        cls,
        rg_path: Path,
        pattern: str,
        path: Path,
        glob: Optional[str],
        output_mode: str,
        ignore_case: bool,
        context_lines: int,
    ) -> str:
        args = [str(rg_path), "--color", "never"]

        if output_mode == "content":
            args.extend(["--line-number", "--with-filename", "--no-heading"])
            if context_lines > 0:
                args.extend(["-C", str(context_lines)])
        elif output_mode == "files_with_matches":
            args.append("--files-with-matches")
        elif output_mode == "count":
            args.extend(["--count", "--with-filename"])
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        if ignore_case:
            args.append("--ignore-case")
        if glob:
            args.extend(["--glob", glob])

        args.extend([pattern, str(path)])
        return "& " + " ".join(cls._quote_powershell(arg) for arg in args)

    @staticmethod
    def _quote_powershell(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _best_effort_output_encoding() -> str:
        if os.name != "nt":
            return "utf-8"

        try:
            import ctypes  # type: ignore

            oem_cp = ctypes.windll.kernel32.GetOEMCP()
            if oem_cp:
                return f"cp{oem_cp}"
        except Exception:
            pass

        return locale.getpreferredencoding(False) or sys.getdefaultencoding()

    @staticmethod
    def _decode_output(data: bytes, encoding: str) -> str:
        try:
            return data.decode(encoding)
        except Exception:
            return data.decode("utf-8", errors="replace")

    @staticmethod
    def _truncate_lines(content: str, head_limit: int) -> str:
        lines = content.splitlines()
        if len(lines) <= head_limit:
            return content
        return "\n".join(lines[:head_limit]) + f"\n... ({len(lines) - head_limit} more lines)"
