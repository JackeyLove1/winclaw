"""Persistent session memory: Markdown files under ``.memory/`` plus ``MEMORY.md`` index."""

from __future__ import annotations

import re
from pathlib import Path

MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200


class MemoryManager:
    """
    Load, build, and save persistent memories across sessions.

    One Markdown file per memory (with YAML-style frontmatter), plus a compact ``MEMORY.md`` index.
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memories: dict[str, dict[str, str]] = {}

    def load_all(self) -> None:
        """Load all individual memory files from disk (excluding MEMORY.md)."""
        self.memories = {}
        if not self.memory_dir.exists():
            return

        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text(encoding="utf-8"))
            if not parsed:
                continue
            name = parsed.get("name", md_file.stem)
            self.memories[name] = {
                "description": parsed.get("description", ""),
                "type": parsed.get("type", "project"),
                "content": parsed.get("content", ""),
                "file": md_file.name,
            }

        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    def load_memory_prompt(self) -> str:
        """Build a memory section for injection into the system prompt."""
        if not self.memories:
            return ""

        sections: list[str] = [
            "# Memories (persistent across sessions)",
            "",
        ]
        for mem_type in MEMORY_TYPES:
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            sections.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                sections.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    sections.append(mem["content"].strip())
                sections.append("")
        return "\n".join(sections)

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """
        Save a memory to disk and update the index.

        Returns a status message.
        """
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        if not safe_name:
            return "Error: invalid memory name"

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {mem_type}\n"
            f"---\n"
            f"{content}\n"
        )
        file_name = f"{safe_name}.md"
        file_path = self.memory_dir / file_name
        file_path.write_text(frontmatter, encoding="utf-8")

        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_name,
        }

        self._rebuild_index()

        try:
            rel = file_path.relative_to(Path.cwd())
        except ValueError:
            rel = file_path
        return f"Saved memory '{name}' [{mem_type}] to {rel}"

    def _rebuild_index(self) -> None:
        """Rebuild MEMORY.md from current in-memory state, capped at MAX_INDEX_LINES."""
        lines = ["# Memory Index", ""]
        memory_index = self.memory_dir / "MEMORY.md"
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        memory_index.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str] | None:
        """Parse ``---`` delimited frontmatter + body content."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result: dict[str, str] = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result
