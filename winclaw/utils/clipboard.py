from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pyperclip
from PIL import Image, ImageGrab


def is_clipboard_available() -> bool:
    """Check if the Pyperclip clipboard is available."""
    try:
        pyperclip.paste()
        return True
    except Exception:
        return False


def grab_image_from_clipboard() -> Image.Image | None:
    """Read an image from the clipboard if possible."""

    payload = ImageGrab.grabclipboard()
    if payload is None:
        return None
    if isinstance(payload, Image.Image):
        return payload
    return _open_first_image(payload)


def _open_first_image(paths: Iterable[os.PathLike[str] | str]) -> Image.Image | None:
    for item in paths:
        try:
            path = Path(item)
        except (TypeError, ValueError):
            continue
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.load()
                return img.copy()
        except Exception:
            continue
    return None
