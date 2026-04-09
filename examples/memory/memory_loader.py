"""Factory for the on-disk memory store."""

from pathlib import Path

from examples.memory.memory_manager import MemoryManager


def create_memory_manager(workdir: Path) -> MemoryManager:
    """Memory store under ``<workdir>/.memory``."""
    return MemoryManager(workdir / ".memory")
