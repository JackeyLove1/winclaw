"""Tool registry with structured execution and observability hooks."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from winclaw.tools.base import Tool, ToolError, ToolReturnValue


class ToolExecutionRecord(BaseModel):
    """Structured execution record for logging, debugging, and future tracing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    params: dict[str, Any]
    duration_ms: int
    result: ToolReturnValue

    def render_text(self, analyze_hint: str = "") -> str:
        text = self.result.render_for_model()
        if self.result.is_error and analyze_hint and analyze_hint not in text:
            return text + analyze_hint
        return text


class ToolRegistry:
    """
    Registry for agent tools.

    It owns registration, public tool definitions, and normalized execution output.
    """

    _ANALYZE_HINT = "\n\n[Analyze the error above and try a different approach.]"

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a tool and validate its public definition eagerly."""
        definition = tool.definition

        existing = self._tools.get(definition.name)
        if existing is not None and existing is not tool and not replace:
            raise ValueError(f"Tool '{definition.name}' is already registered")

        if existing is not None and existing is not tool:
            logger.warning("Replacing registered tool '{}'", definition.name)
        else:
            logger.debug("Registering tool '{}'", definition.name)

        self._tools[definition.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if self._tools.pop(name, None) is not None:
            logger.debug("Unregistered tool '{}'", name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in provider schema format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute_with_record(self, name: str, params: dict[str, Any]) -> ToolExecutionRecord:
        """Execute a tool and return a structured record with timing metadata."""
        tool = self._tools.get(name)
        if tool is None:
            result = ToolError(
                message=f"Tool '{name}' not found. Available: {', '.join(self.tool_names)}",
                brief=f"Tool '{name}' not found",
                output=f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}",
                extras={"error_kind": "tool_not_found"},
            )
            return ToolExecutionRecord(tool_name=name, params=params, duration_ms=0, result=result)

        start = perf_counter()
        normalized_params = tool.cast_params(params)

        issues = tool.validate_param_issues(normalized_params)
        if issues:
            duration_ms = int((perf_counter() - start) * 1000)
            message = "; ".join(issue.render() for issue in issues)
            logger.warning("Invalid parameters for tool '{}': {}", name, message)
            result = ToolError(
                message=f"Invalid parameters for tool '{name}': {message}",
                brief=f"Invalid parameters for '{name}'",
                output=f"Error: Invalid parameters for tool '{name}': {message}",
                extras={
                    "error_kind": "invalid_parameters",
                    "validation_error_count": len(issues),
                },
            )
            return ToolExecutionRecord(
                tool_name=name,
                params=normalized_params,
                duration_ms=duration_ms,
                result=result.with_extras(tool_name=name, duration_ms=duration_ms),
            )

        try:
            raw_result = await tool.execute(**normalized_params)
            duration_ms = int((perf_counter() - start) * 1000)
            result = self._normalize_result(name, raw_result).with_extras(
                tool_name=name,
                duration_ms=duration_ms,
            )
            logger.info("Tool '{}' completed in {} ms", name, duration_ms)
            return ToolExecutionRecord(
                tool_name=name,
                params=normalized_params,
                duration_ms=duration_ms,
                result=result,
            )
        except Exception as exc:
            duration_ms = int((perf_counter() - start) * 1000)
            logger.exception("Tool '{}' crashed with params {}", name, normalized_params)
            result = ToolError(
                message=f"Error executing {name}: {exc}",
                brief=f"Tool '{name}' failed",
                output=f"Error executing {name}: {exc}",
                extras={
                    "error_kind": "execution_exception",
                    "exception_type": type(exc).__name__,
                },
            ).with_extras(tool_name=name, duration_ms=duration_ms)
            return ToolExecutionRecord(
                tool_name=name,
                params=normalized_params,
                duration_ms=duration_ms,
                result=result,
            )

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool and return the legacy text response used by the agent loop."""
        record = await self.execute_with_record(name, params)
        return record.render_text(self._ANALYZE_HINT)

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @staticmethod
    def _normalize_result(name: str, result: str | ToolReturnValue) -> ToolReturnValue:
        if isinstance(result, ToolReturnValue):
            return result
        if isinstance(result, str) and result.startswith("Error"):
            return ToolError(message=result, brief=f"Tool '{name}' returned an error", output=result)
        return ToolReturnValue(
            is_error=False,
            output=result,
            message="",
            display=[],
        )
