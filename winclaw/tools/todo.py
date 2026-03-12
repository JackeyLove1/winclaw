"""Todo tool for tracking multi-step progress."""

from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

from winclaw.tools.base import Tool
from winclaw.utils.helpers import get_workspace_path


class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoItem(BaseModel):
    id: str
    status: TodoStatus = TodoStatus.PENDING
    text: str


class TodoTool(Tool):
    """Tool to create, update and delete todo items."""

    def __init__(self):
        self._workspace = get_workspace_path()
        self.todo_items: list[TodoItem] = []

    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return "Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done. Prefer tools over prose."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of todo items to create or update",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique id for the task"},
                            "text": {"type": "string", "description": "Task description"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of the task",
                            },
                        },
                        "required": ["id", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        }

    @property
    def get_todo_list(self) -> list[TodoItem]:
        return self.todo_items

    def get_todo_list_desc(self) -> str:
        if not self.todo_items:
            return "No todos."
        lines = []
        for item in self.todo_items:
            marker = {
                TodoStatus.PENDING: "[ ]",
                TodoStatus.IN_PROGRESS: "[>]",
                TodoStatus.COMPLETED: "[x]",
            }[item.status]
            lines.append(f"{marker} #{item.id}: {item.text}")
        done = sum(1 for item in self.todo_items if item.status == TodoStatus.COMPLETED)
        lines.append(f"\n({done}/{len(self.todo_items)} completed)")
        return "\n".join(lines)

    async def execute(self, items: list[dict[str, Any]], **kwargs: Any) -> str:
        logger.debug("Executing TodoTool with items: {}", items)
        try:
            updated_items: list[TodoItem] = [TodoItem.model_validate(item) for item in items]
            item_map = {item.id: item for item in self.todo_items}

            for item in updated_items:
                item_map[item.id] = item

            self.todo_items = list(item_map.values())
            return self.get_todo_list_desc()
        except ValidationError as exc:
            return f"Error: Invalid todo item payload: {exc}"
