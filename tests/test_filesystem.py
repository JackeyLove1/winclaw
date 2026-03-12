import asyncio

import pytest

from winclaw.tools.filesystem import GrepTool


class MockProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False
        self.wait_called = False

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        self.wait_called = True
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.mark.asyncio
async def test_grep_tool_runs_bundled_rg_via_powershell(monkeypatch, tmp_path):
    calls = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return MockProcess(stdout=b"sample.py:3:needle\r\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "rg.exe").write_bytes(b"")
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path, bin_path=bin_dir)
    result = await tool.execute(pattern="needle", path=str(tmp_path))

    assert result == "sample.py:3:needle"
    assert calls["args"][:5] == (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    )
    assert str(bin_dir / "rg.exe") in calls["args"][5]
    assert "'needle'" in calls["args"][5]
    assert calls["kwargs"]["cwd"] == str(tmp_path)
    assert calls["kwargs"]["stdout"] is asyncio.subprocess.PIPE
    assert calls["kwargs"]["stderr"] is asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_grep_tool_returns_friendly_no_match_message(monkeypatch, tmp_path):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return MockProcess(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "rg.exe").write_bytes(b"")
    (tmp_path / "sample.py").write_text("haystack\n", encoding="utf-8")

    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path, bin_path=bin_dir)
    result = await tool.execute(pattern="needle", path=str(tmp_path))

    assert result == "No matches found for pattern: needle"
