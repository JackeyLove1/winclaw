"""
agent-loop + more tools
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)


uv run python examples/agent2.py
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

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

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
WORKDIR = Path(__file__).resolve().parent
print(f"WORK DIR: {WORKDIR}")


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
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


def run_read(path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read file lines with 1-indexed LINE_NUMBER|LINE_CONTENT output."""
    try:
        text = safe_path(path).read_text()
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


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("offset"), kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read file contents from the workspace. Each line is "
            "'LINE_NUMBER|LINE_CONTENT' (1-indexed). For large files use offset and limit "
            "to read in chunks. Parallel calls can read different files."
        ),
        "input_schema": {
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
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
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
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                inp = block.input
                if block.name == "bash":
                    print(f"\033[33m$ {inp.get('command', '')}\033[0m")
                else:
                    print(f"\033[33m{block.name} {inp}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                if handler is None:
                    output = f"Error: unknown tool {block.name}"
                else:
                    output = handler(**inp)
                print((output or "")[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36magent1 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
