from __future__ import annotations

from typing import Any

import pytest

from winclaw.tools.base import (
    BriefDisplayBlock,
    Tool,
    ToolDefinition,
    ToolOk,
    ToolReturnValue,
    UnknownDisplayBlock,
)
from winclaw.tools.registry import ToolRegistry


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy tool for tests"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "enabled": {"type": "boolean"},
            },
            "required": ["count"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"count={kwargs['count']}, enabled={kwargs.get('enabled')}"


class StructuredTool(Tool):
    @property
    def name(self) -> str:
        return "structured"

    @property
    def description(self) -> str:
        return "Returns a structured tool result"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

    async def execute(self, **kwargs: Any) -> ToolReturnValue:
        return ToolOk(output=kwargs["message"], brief="done")


def test_tool_definition_requires_object_schema() -> None:
    with pytest.raises(ValueError, match="type='object'"):
        ToolDefinition(
            name="bad",
            description="bad schema",
            parameters={"type": "array"},
        )


def test_display_block_unknown_type_falls_back_cleanly() -> None:
    value = ToolReturnValue.model_validate(
        {
            "is_error": False,
            "output": "ok",
            "message": "",
            "display": [
                {"type": "brief", "text": "hello"},
                {"type": "custom-card", "title": "debug"},
            ],
        }
    )

    assert isinstance(value.display[0], BriefDisplayBlock)
    assert isinstance(value.display[1], UnknownDisplayBlock)
    assert value.display[1].original_type == "custom-card"
    assert value.brief == "hello"


@pytest.mark.asyncio
async def test_registry_casts_and_validates_params() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())

    record = await registry.execute_with_record("dummy", {"count": "2", "enabled": "yes"})

    assert record.result.is_error is False
    assert record.params == {"count": 2, "enabled": True}
    assert record.result.render_for_model() == "count=2, enabled=True"
    assert record.result.extras is not None
    assert record.result.extras["tool_name"] == "dummy"


@pytest.mark.asyncio
async def test_registry_returns_structured_validation_errors() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())

    text = await registry.execute("dummy", {"count": 0, "extra": "nope"})

    assert "Error: Invalid parameters for tool 'dummy'" in text
    assert "parameter.count: must be >= 1" in text
    assert "parameter.extra: is not allowed" in text


@pytest.mark.asyncio
async def test_registry_accepts_structured_tool_results() -> None:
    registry = ToolRegistry()
    registry.register(StructuredTool())

    record = await registry.execute_with_record("structured", {"message": "hello"})

    assert record.result.is_error is False
    assert record.result.render_for_model() == "hello"
    assert record.result.brief == "done"
