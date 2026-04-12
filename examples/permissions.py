"""Permission governance for example agents: modes, rules, shell validation, workspace trust."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from examples.interactive import prompt_async

# -- Permission modes --
# ``dangerous``: only shell validators (bash/ps) gate ``bash``; all other tools and
# rule-based deny/ask paths auto-allow. Use with care.
MODES = ("default", "plan", "auto", "dangerous")

READ_ONLY_TOOLS = {"read_file", "bash_readonly", "task_list", "task_get"}

# Tools that modify state (aligned with examples/tools names)
WRITE_TOOLS = {"write_file", "edit_file", "bash", "save_memory", "task_create", "task_update"}


# -- Bash security validation --
class BashSecurityValidator:
    """
    Validate bash commands for obviously dangerous patterns.

    First catch a few high-risk patterns, then let the permission pipeline
    decide whether to deny or ask the user.
    """

    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),  # shell metacharacters
        ("sudo", r"\bsudo\b"),  # privilege escalation
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # recursive delete
        ("cmd_substitution", r"\$\("),  # command substitution
        ("ifs_injection", r"\bIFS\s*="),  # IFS manipulation
    ]

    def validate(self, command: str) -> list[tuple[str, str]]:
        """Return (validator_name, pattern) for each failed check; empty if passed."""
        failures: list[tuple[str, str]] = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def is_safe(self, command: str) -> bool:
        """True only if no validators triggered."""
        return len(self.validate(command)) == 0

    def describe_failures(self, command: str) -> str:
        """Human-readable summary of validation failures."""
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)


bash_validator = BashSecurityValidator()


class PowerShellSecurityValidator:
    """
    Validate PowerShell (and common cmd.exe) constructs for high-risk patterns.

    Runs alongside BashSecurityValidator for the ``bash`` tool, since on Windows
    ``shell=True`` often executes cmd/PowerShell while the tool name stays ``bash``.
    """

    _PS_FLAGS = re.IGNORECASE | re.MULTILINE

    VALIDATORS: list[tuple[str, str]] = [
        ("runas_elevation", r"-Verb\s+runAs"),
        ("remove_item_recurse", r"\b(Remove-Item|ri)\b.*?(?:-Recurse|\s+-r\b)"),
        ("cmd_rd_recursive", r"\b(rd|rmdir)\s+/s\b"),
        ("cmd_del_recursive", r"\bdel(\.exe)?\s+/s\b"),
        ("format_volume", r"\bFormat-Volume\b"),
        ("clear_disk", r"\bClear-Disk\b"),
        ("stop_computer", r"\bStop-Computer\b"),
        ("restart_computer", r"\bRestart-Computer\b"),
        ("shutdown", r"\bshutdown(\.exe)?\b"),
        ("invoke_expression", r"\b(Invoke-Expression|iex)\b"),
    ]

    def validate(self, command: str) -> list[tuple[str, str]]:
        failures: list[tuple[str, str]] = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command, self._PS_FLAGS):
                failures.append((name, pattern))
        return failures

    def is_safe(self, command: str) -> bool:
        return len(self.validate(command)) == 0

    def describe_failures(self, command: str) -> str:
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)


powershell_validator = PowerShellSecurityValidator()

# Names that always deny (no prompt), for combined bash + PowerShell checks.
_SHELL_DENY_VALIDATOR_NAMES = frozenset(
    {
        "sudo",
        "rm_rf",
        "runas_elevation",
        "remove_item_recurse",
        "cmd_rd_recursive",
        "cmd_del_recursive",
        "format_volume",
        "clear_disk",
        "stop_computer",
        "restart_computer",
        "shutdown",
    }
)


def _describe_validated_failures(failures: list[tuple[str, str]]) -> str:
    parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
    return "Security flags: " + ", ".join(parts)


# -- Workspace trust --
def is_workspace_trusted(workspace: Path | None) -> bool:
    """
    True if the workspace has been explicitly marked as trusted via marker file.

    Uses `.claude/.claude_trusted` under the workspace root.
    If workspace is None, returns False.
    """
    if workspace is None:
        return False
    trust_marker = workspace / ".claude" / ".claude_trusted"
    return trust_marker.exists()


# -- Permission rules --
# Rules are checked in order for their behavior class; deny rules run first in check().
# Format: {"tool": "<tool_name_or_*>", "path": "<glob_or_*>", "behavior": "allow|deny|ask"}
DEFAULT_RULES: list[dict[str, Any]] = [
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    {"tool": "bash", "content": "*Format-Volume*", "behavior": "deny"},
    {"tool": "bash", "content": "*Clear-Disk*", "behavior": "deny"},
    {"tool": "bash", "content": "*Stop-Computer*", "behavior": "deny"},
    {"tool": "bash", "content": "*Restart-Computer*", "behavior": "deny"},
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]


class PermissionManager:
    """
    Permission decisions for tool calls.

    Pipeline: shell validation (bash tool) -> deny_rules -> mode_check -> allow_rules -> ask_user
    """

    def __init__(self, mode: str = "dangerous", rules: list[dict[str, Any]] | None = None):
        if mode not in MODES:
            msg = f"Unknown mode: {mode}. Choose from {MODES}"
            raise ValueError(msg)
        self.mode = mode
        self.rules: list[dict[str, Any]] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, str]:
        """Return {"behavior": "allow"|"deny"|"ask", "reason": str}."""
        if tool_name == "bash":
            command = tool_input.get("command", "")
            failures = bash_validator.validate(command)
            failures.extend(powershell_validator.validate(command))
            if failures:
                severe_hits = [f for f in failures if f[0] in _SHELL_DENY_VALIDATOR_NAMES]
                desc = _describe_validated_failures(failures)
                if severe_hits:
                    return {"behavior": "deny", "reason": f"Shell validator: {desc}"}
                return {"behavior": "ask", "reason": f"Shell validator flagged: {desc}"}

        if self.mode == "dangerous":
            return {
                "behavior": "allow",
                "reason": "Dangerous mode: auto-approved (only shell validators apply to bash)",
            }

        for rule in self.rules:
            if rule["behavior"] != "deny":
                continue
            if self._matches(rule, tool_name, tool_input):
                return {"behavior": "deny", "reason": f"Blocked by deny rule: {rule}"}

        if self.mode == "plan":
            if tool_name in WRITE_TOOLS:
                return {"behavior": "deny", "reason": "Plan mode: write operations are blocked"}
            return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}

        if self.mode == "auto":
            if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
                return {"behavior": "allow", "reason": "Auto mode: read-only tool auto-approved"}

        for rule in self.rules:
            if rule["behavior"] != "allow":
                continue
            if self._matches(rule, tool_name, tool_input):
                self.consecutive_denials = 0
                return {"behavior": "allow", "reason": f"Matched allow rule: {rule}"}

        return {"behavior": "ask", "reason": f"No rule matched for {tool_name}, asking user"}

    def ask_user(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Interactive approval. Returns True if approved."""
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        print(f"\n  [Permission] {tool_name}: {preview}")
        try:
            answer = input("  Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        return self._apply_user_answer(tool_name, answer)

    async def ask_user_async(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Async interactive approval for agents already running an event loop."""
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        print(f"\n  [Permission] {tool_name}: {preview}")
        try:
            answer = await prompt_async(
                "<b fg='ansiyellow'>Allow?</b> (y/n/always): ",
                "  Allow? (y/n/always): ",
            )
        except KeyboardInterrupt:
            return False

        return self._apply_user_answer(tool_name, answer)

    def _apply_user_answer(self, tool_name: str, answer: str) -> bool:
        answer = answer.strip().lower()

        if answer == "always":
            self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
            self.consecutive_denials = 0
            return True
        if answer in ("y", "yes"):
            self.consecutive_denials = 0
            return True

        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(
                f"  [{self.consecutive_denials} consecutive denials -- "
                "consider switching to plan mode]"
            )
        return False

    def _matches(self, rule: dict[str, Any], tool_name: str, tool_input: dict[str, Any]) -> bool:
        if rule.get("tool") and rule["tool"] != "*":
            if rule["tool"] != tool_name:
                return False
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path", "")
            if not fnmatch.fnmatch(path, rule["path"]):
                return False
        if "content" in rule:
            command = tool_input.get("command", "")
            if not fnmatch.fnmatch(command, rule["content"]):
                return False
        return True
