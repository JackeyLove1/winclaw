"""
Anthropic Messages API error recovery: exponential backoff, compaction, tool overflow.
Mirrors agents/s11_error_recovery.py patterns.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from anthropic import Anthropic, APIConnectionError, APITimeoutError, RateLimitError

from examples.compact.auto_compact import summary_messages
from examples.compact.micro_compact import micro_compact
from examples.constants import (
    AUTO_COMPACTION_TRIGGER_TOKENS,
    MAX_RETRY_COUNT,
    PERSIST_THRESHOLD,
)
from examples.utils import estimate_context_usage, persist_large_output

MAX_RECOVERY_ATTEMPTS = MAX_RETRY_COUNT
MAX_OUTPUT_RECOVERY_ATTEMPTS = 3
BACKOFF_BASE_DELAY = 1.0
BACKOFF_MAX_DELAY = 60.0
CONTINUATION_MESSAGE = (
    "Your previous reply stopped because the output length limit was reached. "
    "Continue exactly from where you left off; do not repeat text already produced."
)
POST_TOOL_TOKEN_THRESHOLD = AUTO_COMPACTION_TRIGGER_TOKENS


def _backoff_delay(attempt: int) -> float:
    exp = min(BACKOFF_BASE_DELAY * (2**attempt), BACKOFF_MAX_DELAY)
    return exp + random.random()


def _error_text(exc: BaseException) -> str:
    body = str(exc).lower()
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body += " " + str(resp.text).lower()
        except Exception:
            pass
    return body


def _is_prompt_too_long_error(exc: BaseException) -> bool:
    body = _error_text(exc)
    if "overlong_prompt" in body:
        return True
    if "prompt" in body and "long" in body:
        return True
    if "context_length" in body or "context length" in body:
        return True
    if "too many tokens" in body or "token limit" in body:
        return True
    if "exceeds the" in body and "token" in body:
        return True
    return False


def _is_tool_result_too_long_error(exc: BaseException) -> bool:
    body = _error_text(exc)
    if "tool_result" in body and ("long" in body or "large" in body or "limit" in body):
        return True
    if "tool use" in body and "content" in body and ("long" in body or "too large" in body):
        return True
    return False


def _is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if status is not None and status >= 500:
        return True
    return False


async def _recover_prompt_too_long(
    client: Anthropic,
    messages: list[Any],
    model: str,
) -> None:
    print("[Recovery] Prompt or context too long. Compacting conversation…")
    try:
        compacted = await asyncio.to_thread(
            summary_messages,
            messages,
            client=client,
            model=model,
        )
        messages[:] = micro_compact(compacted)
    except Exception as e:
        print(f"[Recovery] Compaction failed: {e}")
        messages[:] = [
            {
                "role": "user",
                "content": (
                    "Automatic context compaction failed "
                    f"({e!s}). Ask the user for a shorter goal or restate the task."
                ),
            }
        ]


async def _recover_tool_result_overflow(
    client: Anthropic,
    messages: list[Any],
    model: str,
) -> None:
    """Aggressively shrink tool outputs in history, then fall back to session summary."""
    print("[Recovery] Tool result payload likely too large. Pruning tool results…")

    def _shrink_block(obj: Any) -> Any:
        if isinstance(obj, dict) and obj.get("type") == "tool_result":
            content = obj.get("content", "")
            if isinstance(content, str) and len(content) > PERSIST_THRESHOLD:
                tid = str(obj.get("tool_use_id", "unknown"))
                obj = {
                    **obj,
                    "content": persist_large_output(tid, content),
                }
        return obj

    def _shrink_message(msg: Any) -> Any:
        if not isinstance(msg, dict):
            return msg
        content = msg.get("content")
        if isinstance(content, list):
            msg = {**msg, "content": [_shrink_block(b) for b in content]}
        return msg

    try:
        messages[:] = [_shrink_message(m) for m in messages]
        messages[:] = micro_compact(messages)
    except Exception as e:
        print(f"[Recovery] In-place tool shrink failed: {e}")
    usage = estimate_context_usage(messages)
    if usage["estimated_tokens"] >= AUTO_COMPACTION_TRIGGER_TOKENS:
        await _recover_prompt_too_long(client, messages, model=model)


async def create_messages_resilient(
    client: Anthropic,
    *,
    messages: list[Any],
    system: str,
    tools: list[Any],
    model: str,
    max_tokens: int,
) -> Any:
    attempt = 0
    while True:
        try:
            return await asyncio.to_thread(
                client.messages.create,
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if _is_prompt_too_long_error(e):
                await _recover_prompt_too_long(client, messages, model=model)
                continue
            if _is_tool_result_too_long_error(e):
                await _recover_tool_result_overflow(client, messages, model=model)
                continue
            if _is_transient_api_error(e):
                if attempt < MAX_RECOVERY_ATTEMPTS:
                    delay = _backoff_delay(attempt)
                    print(
                        f"[Recovery] Transient API error ({e!s}). "
                        f"Retry in {delay:.1f}s ({attempt + 1}/{MAX_RECOVERY_ATTEMPTS})"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                print(f"[Error] API failed after {MAX_RECOVERY_ATTEMPTS} retries: {e}")
                raise
            raise
