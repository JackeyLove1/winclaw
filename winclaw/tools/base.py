"""
Core tool contracts and shared validation utilities.

This module keeps the tool layer split into a few stable responsibilities:

1. `ToolDefinition` validates the public schema exposed to the model.
2. `Tool` provides schema-driven parameter casting and validation.
3. `ToolReturnValue` provides a structured execution result for richer UX/debugging.
4. `DisplayBlock` provides an extensible user-facing display protocol.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from winclaw.utils.typing import JsonType

try:
    import jsonschema
except ImportError:
    jsonschema = None

ContentPart: TypeAlias = dict[str, JsonType]
ParametersSchema: TypeAlias = dict[str, Any]
ToolOutput: TypeAlias = str | list[ContentPart]


# TODO: support anthropic tool call
class ToolDefinition(BaseModel):
    """Validated public definition for a tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: ParametersSchema

    @model_validator(mode="after")
    def _validate_parameters(self) -> ToolDefinition:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty")

        if not self.description.strip():
            raise ValueError(f"Tool '{self.name}' description cannot be empty")

        if not isinstance(self.parameters, dict):
            raise ValueError(f"Tool '{self.name}' parameters must be a JSON object schema")

        if self.parameters.get("type", "object") != "object":
            raise ValueError(
                f"Tool '{self.name}' parameters schema must have type='object', "
                f"got {self.parameters.get('type')!r}"
            )

        properties = self.parameters.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Tool '{self.name}' schema 'properties' must be an object")

        required = self.parameters.get("required", [])
        if required is not None and not isinstance(required, list):
            raise ValueError(f"Tool '{self.name}' schema 'required' must be a list")

        if jsonschema is not None:
            jsonschema.Draft202012Validator.check_schema(self.parameters)

        return self

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert the definition to the function-tool format used by providers."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolValidationIssue(BaseModel):
    """Structured validation issue that can be logged or rendered."""

    path: str = "parameter"
    message: str

    def render(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


class DisplayBlock(BaseModel, ABC):
    """
    Extensible user-facing display block.

    Subclasses register themselves by `type`, which allows unknown block payloads
    to degrade gracefully instead of breaking deserialization.
    """

    model_config = ConfigDict(extra="allow")

    __display_block_registry: ClassVar[dict[str, type["DisplayBlock"]]] = {}

    type: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "DisplayBlock":
            return

        type_value = getattr(cls, "type", None)
        if type_value is None or not isinstance(type_value, str) or not type_value.strip():
            raise ValueError(
                f"DisplayBlock subclass {cls.__name__} must declare a non-empty string `type`"
            )
        cls.__display_block_registry[type_value] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if cls.__name__ != "DisplayBlock":
            return handler(source_type)

        def validate_display_block(value: Any) -> Any:
            if hasattr(value, "__class__") and issubclass(value.__class__, cls):
                return value

            if isinstance(value, dict) and "type" in value:
                raw_type = cast(dict[str, Any], value).get("type")
                if not isinstance(raw_type, str):
                    raise ValueError(f"Cannot validate {value!r} as DisplayBlock")

                target_class = cls.__display_block_registry.get(raw_type)
                if target_class is None:
                    payload = {
                        key: item
                        for key, item in cast(dict[str, Any], value).items()
                        if key != "type"
                    }
                    return UnknownDisplayBlock.model_validate(
                        {"type": "unknown", "original_type": raw_type, "data": payload}
                    )
                return target_class.model_validate(value)

            raise ValueError(f"Cannot validate {value!r} as DisplayBlock")

        return core_schema.no_info_plain_validator_function(validate_display_block)


class UnknownDisplayBlock(DisplayBlock):
    """Fallback block used when an unknown display type is received."""

    type: str = "unknown"
    original_type: str
    data: JsonType


class BriefDisplayBlock(DisplayBlock):
    """Minimal text block that can be surfaced directly in the UI."""

    type: str = "brief"
    text: str


class ToolReturnValue(BaseModel):
    """Structured tool result used for success, error, and observability metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_error: bool
    output: ToolOutput = ""
    message: str = ""
    display: list[DisplayBlock] = Field(default_factory=list)
    extras: dict[str, JsonType] | None = None

    @property
    def brief(self) -> str:
        for block in self.display:
            if isinstance(block, BriefDisplayBlock):
                return block.text
        return ""

    def output_text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        return "\n".join(json.dumps(part, ensure_ascii=False) for part in self.output)

    def render_for_model(self) -> str:
        output = self.output_text().strip()
        if output:
            return output
        return self.message

    def with_extras(self, **extras: JsonType) -> ToolReturnValue:
        merged = dict(self.extras or {})
        merged.update(extras)
        return self.model_copy(update={"extras": merged})


class ToolOk(ToolReturnValue):
    """Convenience result for successful tool calls."""

    def __init__(
        self,
        *,
        output: str | ContentPart | list[ContentPart],
        message: str = "",
        brief: str = "",
        display: list[DisplayBlock] | None = None,
        extras: dict[str, JsonType] | None = None,
    ) -> None:
        normalized_output: ToolOutput = [output] if isinstance(output, dict) else output
        blocks = list(display or [])
        if brief:
            blocks.insert(0, BriefDisplayBlock(text=brief))
        super().__init__(
            is_error=False,
            output=normalized_output,
            message=message,
            display=blocks,
            extras=extras,
        )


class ToolError(ToolReturnValue):
    """Convenience result for failed tool calls."""

    def __init__(
        self,
        *,
        message: str,
        brief: str,
        output: str | ContentPart | list[ContentPart] = "",
        display: list[DisplayBlock] | None = None,
        extras: dict[str, JsonType] | None = None,
    ) -> None:
        normalized_output: ToolOutput = [output] if isinstance(output, dict) else output
        blocks = list(display or [])
        if brief:
            blocks.insert(0, BriefDisplayBlock(text=brief))
        super().__init__(
            is_error=True,
            output=normalized_output,
            message=message,
            display=blocks,
            extras=extras,
        )


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Subclasses only need to define the public schema and implement `execute`.
    The base class handles parameter coercion and validation in a predictable way.
    """

    _TYPE_MAP: ClassVar[dict[str, type[Any] | tuple[type[Any], ...]]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> ParametersSchema:
        """JSON Schema for tool parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | ToolReturnValue:
        """Execute the tool with the given parameters."""

    @cached_property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply safe schema-driven casts before validation."""
        schema = self.definition.parameters
        if schema.get("type", "object") != "object" or not isinstance(params, dict):
            return params
        return cast(dict[str, Any], self._cast_object(params, schema))

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate tool parameters against the tool JSON schema."""
        return [issue.render() for issue in self.validate_param_issues(params)]

    def validate_param_issues(self, params: dict[str, Any]) -> list[ToolValidationIssue]:
        """Return structured parameter validation issues."""
        if not isinstance(params, dict):
            return [
                ToolValidationIssue(
                    message=f"must be an object, got {type(params).__name__}",
                )
            ]
        return self._validate(params, self.definition.parameters, "parameter")

    def to_schema(self) -> dict[str, Any]:
        """Convert the tool definition to the provider schema format."""
        return self.definition.to_openai_schema()

    def _cast_object(self, obj: Any, schema: ParametersSchema) -> Any:
        if not isinstance(obj, dict):
            return obj

        props = schema.get("properties", {})
        if not isinstance(props, dict):
            return obj

        casted: dict[str, Any] = {}
        for key, value in obj.items():
            child_schema = props.get(key)
            casted[key] = self._cast_value(value, child_schema) if child_schema else value
        return casted

    def _cast_value(self, value: Any, schema: ParametersSchema | None) -> Any:
        if not schema:
            return value

        target_type = schema.get("type")

        if target_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return value

        if target_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return value
            return value

        if target_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            return value

        if target_type == "string":
            return value if value is None else str(value)

        if target_type == "array" and isinstance(value, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                return [self._cast_value(item, item_schema) for item in value]
            return value

        if target_type == "object" and isinstance(value, dict):
            return self._cast_object(value, schema)

        return value

    def _validate(
        self,
        value: Any,
        schema: ParametersSchema,
        path: str,
    ) -> list[ToolValidationIssue]:
        schema_type = schema.get("type")
        issues = self._validate_type(value, schema_type, path)
        if issues:
            return issues

        if "enum" in schema and value not in schema["enum"]:
            issues.append(
                ToolValidationIssue(
                    path=path,
                    message=f"must be one of {schema['enum']!r}, got {value!r}",
                )
            )

        if schema_type in {"integer", "number"}:
            issues.extend(self._validate_numeric(value, schema, path))
        elif schema_type == "string":
            issues.extend(self._validate_string(value, schema, path))
        elif schema_type == "object":
            issues.extend(self._validate_object(value, schema, path))
        elif schema_type == "array":
            issues.extend(self._validate_array(value, schema, path))

        return issues

    def _validate_type(
        self,
        value: Any,
        schema_type: str | None,
        path: str,
    ) -> list[ToolValidationIssue]:
        if schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return [
                    ToolValidationIssue(
                        path=path,
                        message=f"should be integer, got {type(value).__name__} ({value!r})",
                    )
                ]
            return []

        if schema_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return [
                    ToolValidationIssue(
                        path=path,
                        message=f"should be number, got {type(value).__name__} ({value!r})",
                    )
                ]
            return []

        expected = self._TYPE_MAP.get(cast(str, schema_type))
        if expected and not isinstance(value, expected):
            return [
                ToolValidationIssue(
                    path=path,
                    message=f"should be {schema_type}, got {type(value).__name__} ({value!r})",
                )
            ]

        return []

    @staticmethod
    def _validate_numeric(
        value: int | float,
        schema: ParametersSchema,
        path: str,
    ) -> list[ToolValidationIssue]:
        issues: list[ToolValidationIssue] = []
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(
                ToolValidationIssue(
                    path=path, message=f"must be >= {schema['minimum']}, got {value!r}"
                )
            )
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(
                ToolValidationIssue(
                    path=path, message=f"must be <= {schema['maximum']}, got {value!r}"
                )
            )
        return issues

    @staticmethod
    def _validate_string(
        value: str, schema: ParametersSchema, path: str
    ) -> list[ToolValidationIssue]:
        issues: list[ToolValidationIssue] = []
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(
                ToolValidationIssue(
                    path=path,
                    message=f"must be at least {schema['minLength']} chars, got {len(value)}",
                )
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            issues.append(
                ToolValidationIssue(
                    path=path,
                    message=f"must be at most {schema['maxLength']} chars, got {len(value)}",
                )
            )
        return issues

    def _validate_object(
        self,
        value: dict[str, Any],
        schema: ParametersSchema,
        path: str,
    ) -> list[ToolValidationIssue]:
        issues: list[ToolValidationIssue] = []
        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", True)
        strict = schema.get("strict", False)

        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    issues.append(ToolValidationIssue(path=f"{path}.{key}", message="is required"))

        if not isinstance(properties, dict):
            return issues

        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                issues.extend(self._validate(item, properties[key], child_path))
                continue

            if additional_properties is False or strict:
                issues.append(ToolValidationIssue(path=child_path, message="is not allowed"))
                continue

            if isinstance(additional_properties, dict):
                issues.extend(self._validate(item, additional_properties, child_path))

        return issues

    def _validate_array(
        self,
        value: list[Any],
        schema: ParametersSchema,
        path: str,
    ) -> list[ToolValidationIssue]:
        issues: list[ToolValidationIssue] = []

        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(
                ToolValidationIssue(
                    path=path,
                    message=f"must contain at least {schema['minItems']} items, got {len(value)}",
                )
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(
                ToolValidationIssue(
                    path=path,
                    message=f"must contain at most {schema['maxItems']} items, got {len(value)}",
                )
            )

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(self._validate(item, item_schema, f"{path}[{index}]"))

        return issues
