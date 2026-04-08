"""Scan skills/<name>/SKILL.md with YAML frontmatter (--- delimiters)."""

import re
from pathlib import Path
from typing import Any

import yaml

# Frontmatter: opening ---, body, closing ---, then markdown. Tolerate CRLF.
_FRONTMATTER_RE = re.compile(
    r"^---\s*\r?\n(?P<fm>.+?)\r?\n---\s*\r?\n(?P<body>.*)",
    re.DOTALL,
)


class SkillLoader:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name_raw = meta.get("name", f.parent.name)
            if name_raw is None:
                name = f.parent.name
            else:
                s = str(name_raw).strip()
                name = s if s else f.parent.name
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter between --- delimiters."""
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return {}, text
        raw = match.group("fm").strip()
        try:
            meta_obj = yaml.safe_load(raw)
        except yaml.YAMLError:
            meta_obj = None
        if not isinstance(meta_obj, dict):
            meta = {}
        else:
            meta = meta_obj
        return meta, match.group("body").strip()

    @staticmethod
    def _prompt_str(value: Any, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _tags_display(tags: Any) -> str:
        if tags is None:
            return ""
        if isinstance(tags, str):
            return tags
        if isinstance(tags, list):
            return ", ".join(str(t) for t in tags)
        return str(tags)

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            meta = skill["meta"]
            desc = self._prompt_str(meta.get("description"), "No description")
            tags = self._tags_display(meta.get("tags"))
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            avail = ", ".join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {avail}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'

    def reload(self) -> None:
        """Rescan skills directory (e.g. after files change)."""
        self.skills.clear()
        self._load_all()
