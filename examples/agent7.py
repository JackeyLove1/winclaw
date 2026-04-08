"""
agent7: agent5 stack + SkillLoader (examples/skills) + load_skill tool

Overview
    - Parent agent keeps full chat history in the REPL session.
    - Parent tools = CHILD_TOOLS + task + fork.
    - Skills: Layer 1 = skill names/descriptions in system prompts; Layer 2 =
      load_skill(name) returns full SKILL.md body for tool_result.
    - task: subagent with fresh context (same filesystem/tools), summary only.
    - fork: subagent with a deep copy of parent history (excluding the pending
      tool_use turn), so it inherits context; summary only returns to parent.

Startup
    Import-time
        examples/skills/**/SKILL.md --> SkillLoader(SKILLS_DIR)
              |
              +--> SKILL_LOADER.get_descriptions()
                   --> SYSTEM / SUBAGENT_SYSTEM / FORK_SUBAGENT_SYSTEM
              |
              +--> LoadSkillTool(SKILL_LOADER) included in _BASE_TOOLS

    After MCP connect (async_main)
        WORKDIR/mcp.json --> load_mcp_tools_async() --> MCP_TOOLS_CACHE
              |
        _BASE_TOOLS (bash, read/write/edit, load_skill) --+
              |                                            |
              +----------------------+---------------------+
                                     v
              merged_child = _BASE_TOOLS + MCP_TOOLS_CACHE
                                     |
                                     v
                    CHILD_TOOLS (schemas)  +  CHILD_TOOL_BY_NAME (handlers)
                                     |
              +------------------------+------------------------+
              |                                                 |
              v                                                 v
        task / fork inner loops                      parent agent_loop only
        client: tools=CHILD_TOOLS                    TOOLS = CHILD_TOOLS
        (no task / fork in child API)              + task + fork schemas
                                                     TOOL_BY_NAME += TASK_TOOL, FORK_TOOL
        _register_all_tools() rewrites the four globals above.

REPL
    stdin --> history --> agent_loop(history) --> print assistant text

Run
    uv run python examples/agent7.py
"""

import asyncio
import copy
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from examples.base import Tool, ToolResult
from examples.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async
from examples.skill_loader import SkillLoader
from examples.tools import (
    BashTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    ForkTool,
    LoadSkillTool,
    TaskTool,
)

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
SKILLS_DIR = WORKDIR / "skills"
SKILL_LOADER = SkillLoader(SKILLS_DIR)
print(f"WORK DIR: {WORKDIR}")

_SYSTEM_SKILLS_BLOCK = (
    f"Use load_skill to access specialized knowledge before tackling unfamiliar topics.\n\n"
    f"Skills available:\n{SKILL_LOADER.get_descriptions()}"
)
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use bash and workspace tools to solve tasks. "
    "MCP tools from configured servers are prefixed SERVER_NAME__ (see tool list). "
    "Use task for isolated subtasks with fresh context; use fork when the subtask needs prior "
    "conversation context. Act, don't explain.\n\n" + _SYSTEM_SKILLS_BLOCK
)
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. You have fresh context and do not see the parent "
    "conversation history. Complete the delegated task with the provided tools, then return only a "
    "concise final summary for the parent agent.\n\n" + _SYSTEM_SKILLS_BLOCK
)
FORK_SUBAGENT_SYSTEM = (
    f"You are a forked coding subagent at {WORKDIR}. You see the parent's conversation history "
    "before this branch. Use the tools to complete the instruction in the latest user message, "
    "then reply with a concise summary for the main agent only (no meta narration).\n\n"
    + _SYSTEM_SKILLS_BLOCK
)

# Set in agent_loop before executing tool calls so fork can read the parent's message list.
_parent_messages_ctx: ContextVar[list[Any] | None] = ContextVar(
    "_parent_messages_ctx", default=None
)


_BASE_TOOLS: list[Tool] = [
    BashTool(WORKDIR),
    FileReadTool(WORKDIR),
    FileWriteTool(WORKDIR),
    FileEditTool(WORKDIR),
    LoadSkillTool(SKILL_LOADER),
]
# Filled when MCP servers connect (list_tools); keeps Tool instances alive for the session.
MCP_TOOLS_CACHE: list[Tool] = []
CHILD_TOOLS = [t.to_schema() for t in _BASE_TOOLS]
CHILD_TOOL_BY_NAME: dict[str, Tool] = {t.name: t for t in _BASE_TOOLS}


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


TASK_TOOL = TaskTool(run_subagent)
FORK_TOOL = ForkTool(run_fork_agent, lambda: _parent_messages_ctx.get())
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
                query = input("\033[36magent7 >> \033[0m")
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
