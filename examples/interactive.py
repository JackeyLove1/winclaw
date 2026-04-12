"""Interactive prompt helpers for example agents."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

_PROMPT_SESSION: PromptSession | None = None


def _supports_prompt_toolkit() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def init_prompt_session(history_file: Path) -> None:
    """Initialize a reusable prompt session with persistent history."""
    global _PROMPT_SESSION
    if _PROMPT_SESSION is not None or not _supports_prompt_toolkit():
        return

    history_file.parent.mkdir(parents=True, exist_ok=True)
    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,
    )


async def prompt_async(html_prompt: str, plain_prompt: str) -> str:
    """Read interactive input, using prompt_toolkit when a TTY is available."""
    try:
        if _PROMPT_SESSION is not None and _supports_prompt_toolkit():
            with patch_stdout():
                return await _PROMPT_SESSION.prompt_async(HTML(html_prompt))
        return await asyncio.to_thread(input, plain_prompt)
    except EOFError as exc:
        raise KeyboardInterrupt from exc
