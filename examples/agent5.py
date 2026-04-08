"""
agent5: REPL + Anthropic tools + workspace tools + MCP + task + fork dispatcher

Overview
    - Parent agent keeps full chat history in the REPL session.
    - Parent tools = CHILD_TOOLS + task + fork.
    - task: subagent with fresh context (same filesystem/tools), summary only.
    - fork: subagent with a deep copy of parent history (excluding the pending
      tool_use turn), so it inherits context; summary only returns to parent.

Startup
    mcp.json --> load_mcp_tools_async() --> MCP tool wrappers
                                           |
    base tools ----------------------------+
                                           v
                                CHILD_TOOLS / CHILD_TOOL_BY_NAME
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                        subagent tools      parent: task + fork
                                                      |
                                                      v
                                            TOOLS / TOOL_BY_NAME

REPL
    stdin --> history --> agent_loop(history) --> print assistant text

Run
    uv run python examples/agent5.py
"""

import asyncio
import copy
import os
import subprocess
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from examples.base import Tool, ToolResult
from examples.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)
load_dotenv(override=True)


if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

if os.getenv("ANTHROPIC_BASE_URL") is None or os.getenv("ANTHROPIC_API_KEY") is None:
    raise ValueError("ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY must be set")

client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path(__file__).resolve().parent
print(f"WORK DIR: {WORKDIR}")

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Use bash and workspace tools to solve tasks. "
    "MCP tools from configured servers are prefixed SERVER_NAME__ (see tool list). "
    "Use task for isolated subtasks with fresh context; use fork when the subtask needs prior "
    "conversation context. Act, don't explain."
)
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. You have fresh context and do not see the parent "
    "conversation history. Complete the delegated task with the provided tools, then return only a "
    "concise final summary for the parent agent."
)
FORK_SUBAGENT_SYSTEM = (
    f"You are a forked coding subagent at {WORKDIR}. You see the parent's conversation history "
    "before this branch. Use the tools to complete the instruction in the latest user message, "
    "then reply with a concise summary for the main agent only (no meta narration)."
)

# Set in agent_loop before executing tool calls so fork can read the parent's message list.
_parent_messages_ctx: ContextVar[list[Any] | None] = ContextVar("_parent_messages_ctx", default=None)


def _safe_path(workdir: Path, p: str) -> Path:
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _read_workspace_file(path: Path) -> tuple[str, str]:
    """Decode file for workspace tools: prefer UTF-8, then common Chinese legacy encodings."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


class BashTool(Tool):
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Run a shell command."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    def _run_sync(self, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self._workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (120s)"

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        content = await asyncio.to_thread(self._run_sync, kwargs["command"])
        return ToolResult(success=True, content=content)


class ReadFileTool(Tool):
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
            text, _ = _read_workspace_file(_safe_path(self._workdir, path))
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


class WriteFileTool(Tool):
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
            fp = _safe_path(self._workdir, path)
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


class EditFileTool(Tool):
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
            fp = _safe_path(self._workdir, path)
            content, enc = _read_workspace_file(fp)
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


def _response_text(blocks: list[Any]) -> str:
    text_parts = []
    for block in blocks:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)
    return "".join(text_parts).strip()


async def _execute_tool_block(
    block: Any, tool_by_name: dict[str, Tool], log_prefix: str = ""
) -> str:
    inp = block.input
    if block.name == "bash":
        print(f"\033[33m{log_prefix}$ {inp.get('command', '')}\033[0m")
    else:
        print(f"\033[33m{log_prefix}{block.name} {inp}\033[0m")

    agent_tool = tool_by_name.get(block.name)
    if agent_tool is None:
        output = f"Error: unknown tool {block.name}"
    else:
        tool_result = await agent_tool.execute(**inp)
        output = _tool_output_string(tool_result)

    print((output or "")[:200])
    return output


async def run_subagent(prompt: str) -> str:
    sub_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    response = None

    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = await _execute_tool_block(
                    block,
                    CHILD_TOOL_BY_NAME,
                    log_prefix="[subagent] ",
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:50000],
                    }
                )
        sub_messages.append({"role": "user", "content": results})

    if response is None:
        return "(no summary)"
    if response.stop_reason == "tool_use":
        return "Subagent stopped after reaching the tool loop safety limit."
    return _response_text(response.content) or "(no summary)"


def _fork_inherited_messages(parent_messages: list[Any]) -> list[Any]:
    """Copy parent history for a new API call: drop the last assistant turn (pending tool_use)."""
    if len(parent_messages) < 1:
        return []
    return copy.deepcopy(parent_messages[:-1])


async def run_fork_agent(parent_messages: list[Any], prompt: str) -> str:
    sub_messages: list[dict[str, Any]] = _fork_inherited_messages(parent_messages)
    sub_messages.append(
        {
            "role": "user",
            "content": (
                "[Fork — you inherit the conversation above. Complete this branch; end with a concise "
                "summary for the main agent.]\n\n" + prompt
            ),
        }
    )
    response = None

    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=FORK_SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = await _execute_tool_block(
                    block,
                    CHILD_TOOL_BY_NAME,
                    log_prefix="[fork] ",
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:50000],
                    }
                )
        sub_messages.append({"role": "user", "content": results})

    if response is None:
        return "(no summary)"
    if response.stop_reason == "tool_use":
        return "Fork agent stopped after reaching the tool loop safety limit."
    return _response_text(response.content) or "(no summary)"


class TaskTool(Tool):
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
        summary = await run_subagent(kwargs["prompt"])
        return ToolResult(success=True, content=summary)


class ForkTool(Tool):
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
        parent = _parent_messages_ctx.get()
        if parent is None:
            return ToolResult(
                success=False,
                content="",
                error="fork could not read parent messages (internal)",
            )
        summary = await run_fork_agent(parent, kwargs["prompt"])
        return ToolResult(success=True, content=summary)


_BASE_TOOLS: list[Tool] = [
    BashTool(WORKDIR),
    ReadFileTool(WORKDIR),
    WriteFileTool(WORKDIR),
    EditFileTool(WORKDIR),
]
# Filled when MCP servers connect (list_tools); keeps Tool instances alive for the session.
MCP_TOOLS_CACHE: list[Tool] = []
TASK_TOOL = TaskTool()
FORK_TOOL = ForkTool()
CHILD_TOOLS = [t.to_schema() for t in _BASE_TOOLS]
CHILD_TOOL_BY_NAME: dict[str, Tool] = {t.name: t for t in _BASE_TOOLS}
TOOLS = [*CHILD_TOOLS, TASK_TOOL.to_schema(), FORK_TOOL.to_schema()]
TOOL_BY_NAME: dict[str, Tool] = {
    **CHILD_TOOL_BY_NAME,
    TASK_TOOL.name: TASK_TOOL,
    FORK_TOOL.name: FORK_TOOL,
}


def _tool_output_string(r: ToolResult) -> str:
    if r.error:
        return f"Error: {r.error}"
    return r.content or ""


# -- The core pattern: a while loop that calls tools until the model stops --
# Use async execution so MCP ClientSession stays on the same event loop as connect().
async def agent_loop(messages: list) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        _parent_messages_ctx.set(messages)
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = await _execute_tool_block(block, TOOL_BY_NAME)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


async def _init_mcp_tools() -> list[Tool]:
    """Connect MCP servers, list tools once, return cached Tool wrappers."""
    return await load_mcp_tools_async(str(WORKDIR / "mcp.json"))


def _register_all_tools(mcp_tools: list[Tool]) -> None:
    """Merge base + MCP tools into Anthropic tool list and name lookup."""
    global CHILD_TOOLS, CHILD_TOOL_BY_NAME, TOOLS, TOOL_BY_NAME, MCP_TOOLS_CACHE
    MCP_TOOLS_CACHE = mcp_tools
    merged_child = _BASE_TOOLS + MCP_TOOLS_CACHE
    CHILD_TOOLS = [t.to_schema() for t in merged_child]
    CHILD_TOOL_BY_NAME = {t.name: t for t in merged_child}
    TOOLS = [*CHILD_TOOLS, TASK_TOOL.to_schema(), FORK_TOOL.to_schema()]
    TOOL_BY_NAME = {
        **CHILD_TOOL_BY_NAME,
        TASK_TOOL.name: TASK_TOOL,
        FORK_TOOL.name: FORK_TOOL,
    }


async def async_main() -> None:
    """Single event loop for MCP transport + tool calls + shutdown."""
    mcp_tools = await _init_mcp_tools()
    _register_all_tools(mcp_tools)
    history: list = []
    try:
        while True:
            try:
                query = input("\033[36magent5 >> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            history.append({"role": "user", "content": query})
            await agent_loop(history)
            response_content = history[-1]["content"]
            if isinstance(response_content, list):
                for block in response_content:
                    if hasattr(block, "text"):
                        print(block.text)
            print()
    finally:
        await cleanup_mcp_connections()


if __name__ == "__main__":
    asyncio.run(async_main())
