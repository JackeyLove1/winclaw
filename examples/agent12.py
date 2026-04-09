"""
agent12: agent loop + mcp + sub-agent + skills + context compaction
        + permission governance + Hook + memory system
        + SystemPromptBuilder (sectioned system prompt + dynamic boundary)
        + Error Recovery


Run
    uv run python examples/agent12.py
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from examples.base import Tool, ToolResult
from examples.compact.auto_compact import summary_messages
from examples.compact.micro_compact import micro_compact
from examples.constants import (
    AUTO_COMPACTION_TRIGGER_TOKENS,
    CONSOLE_TOOL_OUTPUT_MAX_CHARS,
    CONTEXT_LIMIT,
    MESSAGES_MAX_TOKENS,
    PERSIST_THRESHOLD,
    SUBAGENT_TOOL_LOOP_MAX,
)
from examples.hook import HookManager
from examples.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async
from examples.memory.memory_loader import create_memory_manager
from examples.memory.memory_manager import MemoryManager
from examples.memory.prompt import MEMORY_GUIDANCE
from examples.permissions import MODES, PermissionManager
from examples.recovery import (
    CONTINUATION_MESSAGE,
    MAX_OUTPUT_RECOVERY_ATTEMPTS,
    POST_TOOL_TOKEN_THRESHOLD,
    create_messages_resilient,
)
from examples.skill_loader import SkillLoader
from examples.tools import (
    BashTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    ForkTool,
    LoadSkillTool,
    SaveMemoryTool,
    TaskTool,
)
from examples.utils import (
    estimate_context_usage,
    estimate_tokens,
    get_platform,
    get_skill_dir,
    get_work_dir,
    persist_large_output,
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
FAST_MODEL = os.environ["FAST_MODEL_ID"]
WORKDIR = get_work_dir(__file__)
SKILLS_DIR = get_skill_dir()
SKILL_LOADER = SkillLoader(SKILLS_DIR)
MEMORY_MANAGER = create_memory_manager(WORKDIR)
print(f"WORK DIR: {WORKDIR}")

_perm_mode = os.getenv("WINCLAW_PERMISSION_MODE", "dangerous").strip().lower()
if _perm_mode not in MODES:
    raise ValueError(f"WINCLAW_PERMISSION_MODE must be one of {MODES}, got {_perm_mode!r}")
PERMISSION_MANAGER = PermissionManager(mode=_perm_mode)
print(f"Permission mode: {_perm_mode}")

DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="


class SystemPromptBuilder:
    """
    Assemble the system prompt from independent sections.

    Each section has one source and one responsibility so the prompt stays
    easier to reason about, test, and extend.
    """

    def __init__(
        self,
        workdir: Path,
        skill_loader: SkillLoader,
        memory_manager: MemoryManager,
    ) -> None:
        self.workdir = workdir
        self._skill_loader = skill_loader
        self._memory_manager = memory_manager

    def _build_core(self, core: str) -> str:
        return core.strip()

    def _build_tool_listing(self, tools: list[dict[str, Any]]) -> str:
        if not tools:
            return ""
        lines = ["# Available tools"]
        for tool in tools:
            props = tool.get("input_schema", {}).get("properties", {})
            params = ", ".join(props.keys())
            lines.append(f"- {tool['name']}({params}): {tool['description']}")
        return "\n".join(lines)

    def _build_skill_listing(self) -> str:
        if not self._skill_loader.skills:
            return ""
        lines = ["# Available skills", "Use load_skill to load full skill bodies when needed."]
        lines.append(self._skill_loader.get_descriptions())
        return "\n".join(lines)

    def _build_memory_section(self) -> str:
        content = self._memory_manager.load_memory_prompt()
        return content.strip() if content else ""

    def _build_memory_guidance(self) -> str:
        return "# When to save memories\n" + MEMORY_GUIDANCE.strip()

    def _build_claude_md(self) -> str:
        """
        Load CLAUDE.md files in priority order (all are included):
        1. ~/.claude/CLAUDE.md (user-global)
        2. <workdir>/CLAUDE.md (workspace)
        3. <cwd>/CLAUDE.md when cwd differs from workdir (directory-specific)
        """
        sources: list[tuple[str, str]] = []

        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            sources.append(
                ("user global (~/.claude/CLAUDE.md)", user_claude.read_text(encoding="utf-8"))
            )

        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            sources.append(("workspace (CLAUDE.md)", project_claude.read_text(encoding="utf-8")))

        cwd = Path.cwd()
        if cwd.resolve() != self.workdir.resolve():
            subdir_claude = cwd / "CLAUDE.md"
            if subdir_claude.exists():
                label = f"subdir ({cwd.name}/CLAUDE.md)"
                sources.append((label, subdir_claude.read_text(encoding="utf-8")))

        if not sources:
            return ""
        parts = ["# CLAUDE.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self) -> str:
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {MODEL}",
            f"Platform: {get_platform()}",
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    def build(self, core: str, tools: list[dict[str, Any]]) -> str:
        """
        Assemble the full system prompt. Static sections (before the boundary) are
        separated from dynamic context so a production agent could cache the static
        prefix across turns.
        """
        sections: list[str] = []

        core_text = self._build_core(core)
        if core_text:
            sections.append(core_text)

        tool_block = self._build_tool_listing(tools)
        if tool_block:
            sections.append(tool_block)

        skills = self._build_skill_listing()
        if skills:
            sections.append(skills)

        memory = self._build_memory_section()
        if memory:
            sections.append(memory)

        guidance = self._build_memory_guidance()
        if guidance:
            sections.append(guidance)

        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)

        sections.append(DYNAMIC_BOUNDARY)

        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        return "\n\n".join(sections)


def build_system_reminder(extra: str | None = None) -> dict[str, Any] | None:
    """
    Build a user-role message wrapping per-turn dynamic content.

    Keeps short-lived reminders out of the long-lived system instructions.
    """
    parts: list[str] = []
    if extra:
        parts.append(extra)
    if not parts:
        return None
    content = "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"
    return {"role": "user", "content": content}


SYSTEM_PROMPT_BUILDER = SystemPromptBuilder(WORKDIR, SKILL_LOADER, MEMORY_MANAGER)

SYSTEM_CORE = (
    f"You are a coding agent operating in {WORKDIR}.\n"
    "Use bash and workspace tools to explore, read, write, and edit files.\n"
    "MCP tools from configured servers are prefixed SERVER_NAME__ (see tool list).\n"
    "Use task for isolated subtasks with fresh context; use fork when the subtask needs prior "
    "conversation context. Act, don't explain; verify by reading files rather than guessing."
)
SUBAGENT_CORE = (
    f"You are a coding subagent at {WORKDIR}. You have fresh context and do not see the parent "
    "conversation history. Complete the delegated task with the provided tools, then return only a "
    "concise final summary for the parent agent."
)
FORK_SUBAGENT_CORE = (
    f"You are a forked coding subagent at {WORKDIR}. You see the parent's conversation history "
    "before this branch. Use the tools to complete the instruction in the latest user message, "
    "then reply with a concise summary for the main agent only (no meta narration)."
)


def _system_for_agent(core: str, tools: list[dict[str, Any]]) -> str:
    """Rebuild each turn so new save_memory writes appear on the next model call."""
    return SYSTEM_PROMPT_BUILDER.build(core=core, tools=tools)


# Set in agent_loop before executing tool calls so fork can read the parent's message list.
_parent_messages_ctx: ContextVar[list[Any] | None] = ContextVar(
    "_parent_messages_ctx", default=None
)

# Populated in async_main; subagent/fork loops read the same instance.
HOOK_MANAGER: HookManager | None = None


_BASE_TOOLS: list[Tool] = [
    BashTool(WORKDIR),
    FileReadTool(WORKDIR),
    FileWriteTool(WORKDIR),
    FileEditTool(WORKDIR),
    LoadSkillTool(SKILL_LOADER),
    SaveMemoryTool(MEMORY_MANAGER),
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


def _tool_input_dict(inp: Any) -> dict[str, Any]:
    if isinstance(inp, dict):
        return dict(inp)
    if hasattr(inp, "model_dump"):
        return inp.model_dump()
    return {}


def _prepare_messages(messages: list[Any]) -> list[Any]:
    messages = micro_compact(messages)
    usage = estimate_context_usage(messages)
    if usage["estimated_tokens"] >= AUTO_COMPACTION_TRIGGER_TOKENS:
        messages = summary_messages(messages, client=client, model=MODEL)
        messages = micro_compact(messages)
    return messages


async def _execute_tool_block(
    block: Any,
    tool_by_name: dict[str, Tool],
    log_prefix: str = "",
    hook_manager: HookManager | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    extra_results: list[dict[str, Any]] = []
    inp_raw = block.input
    inp = _tool_input_dict(inp_raw)
    ctx: dict[str, Any] = {
        "tool_name": block.name,
        "tool_input": dict(inp),
    }

    if hook_manager is not None:
        pre = await asyncio.to_thread(hook_manager.run_hooks, "PreToolUse", ctx)
        for msg in pre.get("messages", []):
            extra_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"[Hook message]: {msg}"[:CONTEXT_LIMIT],
                }
            )
        if pre.get("blocked"):
            reason = pre.get("block_reason", "Blocked by hook")
            return f"Tool blocked by PreToolUse hook: {reason}", extra_results
        inp = _tool_input_dict(ctx["tool_input"])

    decision = PERMISSION_MANAGER.check(block.name, inp)
    if decision["behavior"] == "deny":
        return f"Error: {decision['reason']}", extra_results
    if decision["behavior"] == "ask":
        if not PERMISSION_MANAGER.ask_user(block.name, inp):
            return f"Error: permission denied — {decision['reason']}", extra_results

    if block.name == "bash":
        print(f"\033[33m{log_prefix}$ {inp.get('command', '')}\033[0m")
    else:
        print(f"\033[33m{log_prefix}{block.name} {inp_raw}\033[0m")

    agent_tool = tool_by_name.get(block.name)
    if agent_tool is None:
        output = f"Error: unknown tool {block.name}"
    else:
        try:
            tool_result = await agent_tool.execute(**inp, tool_use_id=block.id)
            output = _tool_output_string(tool_result)
        except Exception as e:
            output = f"Error: tool execution failed: {e!s}"

    if len(output) > PERSIST_THRESHOLD:
        output = persist_large_output(block.id, output)

    if hook_manager is not None:
        ctx["tool_output"] = output
        post = await asyncio.to_thread(hook_manager.run_hooks, "PostToolUse", ctx)
        for msg in post.get("messages", []):
            output += f"\n[Hook note]: {msg}"

    print((output or "")[:CONSOLE_TOOL_OUTPUT_MAX_CHARS])
    return output, extra_results


async def run_subagent(prompt: str) -> str:
    sub_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    response = None
    max_output_recovery_count = 0

    for _ in range(SUBAGENT_TOOL_LOOP_MAX):
        sub_messages = _prepare_messages(sub_messages)
        response = await create_messages_resilient(
            client,
            messages=sub_messages,
            system=_system_for_agent(SUBAGENT_CORE, CHILD_TOOLS),
            tools=CHILD_TOOLS,
            model=MODEL,
            max_tokens=MESSAGES_MAX_TOKENS,
        )
        sub_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            max_output_recovery_count += 1
            if max_output_recovery_count <= MAX_OUTPUT_RECOVERY_ATTEMPTS:
                print(
                    f"[subagent recovery] max_tokens "
                    f"({max_output_recovery_count}/{MAX_OUTPUT_RECOVERY_ATTEMPTS})"
                )
                sub_messages.append({"role": "user", "content": CONTINUATION_MESSAGE})
                continue
            return "Subagent stopped: max_tokens recovery exhausted."

        max_output_recovery_count = 0

        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output, extras = await _execute_tool_block(
                    block,
                    CHILD_TOOL_BY_NAME,
                    log_prefix="[subagent] ",
                    hook_manager=HOOK_MANAGER,
                )
                results.extend(extras)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:CONTEXT_LIMIT],
                    }
                )
        sub_messages.append({"role": "user", "content": results})
        if estimate_tokens(sub_messages) > POST_TOOL_TOKEN_THRESHOLD:
            print("[subagent recovery] Post-tool token estimate high; compacting…")
            try:
                compacted = await asyncio.to_thread(
                    summary_messages,
                    sub_messages,
                    client=client,
                    model=MODEL,
                )
                sub_messages[:] = micro_compact(compacted)
            except Exception as e:
                print(f"[subagent recovery] Compaction failed: {e}")

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
    max_output_recovery_count = 0

    for _ in range(SUBAGENT_TOOL_LOOP_MAX):
        sub_messages = _prepare_messages(sub_messages)
        response = await create_messages_resilient(
            client,
            messages=sub_messages,
            system=_system_for_agent(FORK_SUBAGENT_CORE, CHILD_TOOLS),
            tools=CHILD_TOOLS,
            model=MODEL,
            max_tokens=MESSAGES_MAX_TOKENS,
        )
        sub_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            max_output_recovery_count += 1
            if max_output_recovery_count <= MAX_OUTPUT_RECOVERY_ATTEMPTS:
                print(
                    f"[fork recovery] max_tokens "
                    f"({max_output_recovery_count}/{MAX_OUTPUT_RECOVERY_ATTEMPTS})"
                )
                sub_messages.append({"role": "user", "content": CONTINUATION_MESSAGE})
                continue
            return "Fork agent stopped: max_tokens recovery exhausted."

        max_output_recovery_count = 0

        if response.stop_reason != "tool_use":
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output, extras = await _execute_tool_block(
                    block,
                    CHILD_TOOL_BY_NAME,
                    log_prefix="[fork] ",
                    hook_manager=HOOK_MANAGER,
                )
                results.extend(extras)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:CONTEXT_LIMIT],
                    }
                )
        sub_messages.append({"role": "user", "content": results})
        if estimate_tokens(sub_messages) > POST_TOOL_TOKEN_THRESHOLD:
            print("[fork recovery] Post-tool token estimate high; compacting…")
            try:
                compacted = await asyncio.to_thread(
                    summary_messages,
                    sub_messages,
                    client=client,
                    model=MODEL,
                )
                sub_messages[:] = micro_compact(compacted)
            except Exception as e:
                print(f"[fork recovery] Compaction failed: {e}")

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
    max_output_recovery_count = 0
    while True:
        messages = _prepare_messages(messages)
        response = await create_messages_resilient(
            client,
            messages=messages,
            system=_system_for_agent(SYSTEM_CORE, TOOLS),
            tools=TOOLS,
            model=MODEL,
            max_tokens=MESSAGES_MAX_TOKENS,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            max_output_recovery_count += 1
            if max_output_recovery_count <= MAX_OUTPUT_RECOVERY_ATTEMPTS:
                print(
                    f"[Recovery] max_tokens ({max_output_recovery_count}/"
                    f"{MAX_OUTPUT_RECOVERY_ATTEMPTS}). Injecting continuation…"
                )
                messages.append({"role": "user", "content": CONTINUATION_MESSAGE})
                continue
            print(
                f"[Error] max_tokens recovery exhausted ({MAX_OUTPUT_RECOVERY_ATTEMPTS})."
            )
            return

        max_output_recovery_count = 0

        if response.stop_reason != "tool_use":
            return
        _parent_messages_ctx.set(messages)
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output, extras = await _execute_tool_block(
                    block,
                    TOOL_BY_NAME,
                    hook_manager=HOOK_MANAGER,
                )
                results.extend(extras)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:CONTEXT_LIMIT],
                    }
                )
        messages.append({"role": "user", "content": results})
        if estimate_tokens(messages) > POST_TOOL_TOKEN_THRESHOLD:
            print("[Recovery] Token estimate exceeds threshold after tools. Auto-compacting…")
            try:
                compacted = await asyncio.to_thread(
                    summary_messages,
                    messages,
                    client=client,
                    model=MODEL,
                )
                messages[:] = micro_compact(compacted)
            except Exception as e:
                print(f"[Recovery] Post-tool compaction failed: {e}")


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
    global HOOK_MANAGER
    mcp_tools = await _init_mcp_tools()
    _register_all_tools(mcp_tools)
    await asyncio.to_thread(MEMORY_MANAGER.load_all)
    if MEMORY_MANAGER.memories:
        print(f"[{len(MEMORY_MANAGER.memories)} memories loaded into context]")
    else:
        print("[No existing memories. The agent can create them with save_memory.]")
    HOOK_MANAGER = HookManager(WORKDIR)
    await asyncio.to_thread(
        HOOK_MANAGER.run_hooks,
        "SessionStart",
        {"tool_name": "", "tool_input": {}},
    )
    history: list = []
    try:
        while True:
            try:
                query = input("\033[36magent12 >> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            if query.strip() == "/memories":
                if MEMORY_MANAGER.memories:
                    for name, mem in MEMORY_MANAGER.memories.items():
                        print(f"  [{mem['type']}] {name}: {mem['description']}")
                else:
                    print("  (no memories)")
                continue
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
