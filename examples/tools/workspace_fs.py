"""Workspace path safety and file decoding for file tools."""

from pathlib import Path


def safe_workspace_path(workdir: Path, p: str) -> Path:
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def read_workspace_file(path: Path) -> tuple[str, str]:
    """Decode file for workspace tools: prefer UTF-8, then common Chinese legacy encodings."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"
