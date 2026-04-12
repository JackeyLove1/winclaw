from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_toolkit.formatted_text import HTML

from examples import interactive


@pytest.fixture()
def mock_prompt_session():
    session = MagicMock()
    session.prompt_async = AsyncMock()
    with (
        patch("examples.interactive._PROMPT_SESSION", session),
        patch("examples.interactive.patch_stdout"),
        patch("examples.interactive._supports_prompt_toolkit", return_value=True),
    ):
        yield session


@pytest.mark.asyncio
async def test_prompt_async_uses_prompt_toolkit_when_session_is_ready(mock_prompt_session):
    mock_prompt_session.prompt_async.return_value = "hello"

    result = await interactive.prompt_async(
        "<b fg='ansicyan'>agent15&gt;&gt;</b> ",
        "agent15 >> ",
    )

    assert result == "hello"
    mock_prompt_session.prompt_async.assert_called_once()
    args, _ = mock_prompt_session.prompt_async.call_args
    assert isinstance(args[0], HTML)


@pytest.mark.asyncio
async def test_prompt_async_converts_eof_to_keyboard_interrupt(mock_prompt_session):
    mock_prompt_session.prompt_async.side_effect = EOFError()

    with pytest.raises(KeyboardInterrupt):
        await interactive.prompt_async(
            "<b fg='ansicyan'>agent15&gt;&gt;</b> ",
            "agent15 >> ",
        )


@pytest.mark.asyncio
async def test_prompt_async_falls_back_to_plain_input_without_tty():
    with (
        patch("examples.interactive._PROMPT_SESSION", None),
        patch("builtins.input", return_value="exit") as mock_input,
    ):
        result = await interactive.prompt_async(
            "<b fg='ansicyan'>agent15&gt;&gt;</b> ",
            "agent15 >> ",
        )

    assert result == "exit"
    mock_input.assert_called_once_with("agent15 >> ")
