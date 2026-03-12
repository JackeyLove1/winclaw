# TODO: support more format file: pdf, word, excel, etc.
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from winclaw.tools.base import Tool
from winclaw.tools.filesystem import _resolve_path
from winclaw.utils.media import FileType, audio_format_from_mime, detect_file_type

MAX_MEDIA_MEGABYTES = 100
MAX_MEDIA_BYTES = MAX_MEDIA_MEGABYTES * 1024 * 1024
INLINE_DATA_URL_MEGABYTES = 5
INLINE_DATA_URL_BYTES = INLINE_DATA_URL_MEGABYTES * 1024 * 1024
HTTP_TIMEOUT_SECONDS = 20.0


def _to_data_url(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_image_size(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.size
    except Exception:
        return None


def _is_http_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _coerce_file_type(source: str, data: bytes, hinted_mime: str | None = None) -> FileType:
    file_type = detect_file_type(source, data)
    mime_hint = (hinted_mime or "").split(";", 1)[0].strip().lower()
    if file_type.kind == "unknown" and mime_hint.startswith("image/"):
        return FileType(kind="image", mime_type=mime_hint)
    if file_type.kind == "unknown" and mime_hint.startswith("video/"):
        return FileType(kind="video", mime_type=mime_hint)
    if file_type.kind == "unknown" and mime_hint.startswith("audio/"):
        return FileType(kind="audio", mime_type=mime_hint)
    if not file_type.mime_type and mime_hint:
        return FileType(kind=file_type.kind, mime_type=mime_hint)
    return file_type


class Params(BaseModel):
    path: str = Field(
        description="Local file path or http/https URL of the image, video, or audio file to read."
    )
    include_data_url: bool = Field(
        default=False,
        description=(
            "Include a data URL preview for smaller files. Large files are summarized with metadata only."
        ),
    )


class ReadMediaTool(Tool):
    """Read local or remote media metadata."""

    def __init__(self, workspace: Optional[Path] = None, allowed_dir: Optional[Path] = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_media_file"

    @property
    def description(self) -> str:
        return "Read a local path or URL for an image, video, or audio file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local file path or http/https URL of the media file",
                },
                "include_data_url": {
                    "type": "boolean",
                    "description": "Optional: include an inline data URL when the file is small enough",
                },
            },
            "required": ["path"],
        }

    async def _read_local_media(self, path: str) -> tuple[str, str, bytes, FileType]:
        file_path = _resolve_path(path, self._workspace, self._allowed_dir)
        logger.debug("read_media: resolving local path '{}' -> '{}'", path, file_path)
        if not file_path.exists():
            logger.warning("read_media: file not found: {}", path)
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            logger.warning("read_media: not a file (directory): {}", path)
            raise IsADirectoryError(f"Not a file: {path}")
        size = file_path.stat().st_size
        if size > MAX_MEDIA_BYTES:
            logger.warning(
                "read_media: local file too large: {} bytes (limit {} MB), path: {}",
                size,
                MAX_MEDIA_MEGABYTES,
                path,
            )
            raise ValueError(
                f"Media file is too large ({size:,} bytes). Limit is {MAX_MEDIA_MEGABYTES} MB."
            )
        data = file_path.read_bytes()
        file_type = _coerce_file_type(str(file_path), data)
        logger.debug(
            "read_media: read local file {} bytes, kind={} mime={}",
            len(data),
            file_type.kind,
            file_type.mime_type or "unknown",
        )
        return "file", str(file_path), data, file_type

    async def _read_remote_media(self, url: str) -> tuple[str, str, bytes, FileType]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            logger.warning("read_media: unsupported URL scheme or missing netloc: {}", url)
            raise ValueError(f"Unsupported media URL: {url}")

        logger.debug("read_media: fetching remote URL: {}", url)
        async with httpx.AsyncClient(follow_redirects=True, timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_MEDIA_BYTES:
            logger.warning(
                "read_media: remote content-length too large: {} bytes (limit {} MB), url: {}",
                int(content_length),
                MAX_MEDIA_MEGABYTES,
                url,
            )
            raise ValueError(
                f"Remote media is too large ({int(content_length):,} bytes). "
                f"Limit is {MAX_MEDIA_MEGABYTES} MB."
            )

        data = response.content
        if len(data) > MAX_MEDIA_BYTES:
            logger.warning(
                "read_media: remote body too large: {} bytes (limit {} MB), url: {}",
                len(data),
                MAX_MEDIA_MEGABYTES,
                url,
            )
            raise ValueError(
                f"Remote media is too large ({len(data):,} bytes). Limit is {MAX_MEDIA_MEGABYTES} MB."
            )

        hinted_mime = response.headers.get("content-type")
        file_type = _coerce_file_type(url, data, hinted_mime)
        logger.debug(
            "read_media: fetched remote {} bytes, kind={} mime={}",
            len(data),
            file_type.kind,
            file_type.mime_type or "unknown",
        )
        return "url", url, data, file_type

    async def _read_media_file(self, path: str, include_data_url: bool = False) -> str:
        logger.info(
            "read_media: path='{}' include_data_url={}",
            path,
            include_data_url,
        )
        source_kind, resolved_source, data, file_type = (
            await self._read_remote_media(path)
            if _is_http_url(path)
            else await self._read_local_media(path)
        )

        if file_type.kind not in {"image", "video", "audio"}:
            logger.warning(
                "read_media: unsupported media kind '{}' for source: {}",
                file_type.kind,
                resolved_source,
            )
            return (
                "Error: Unsupported media type. "
                "Only image, video, and audio files or URLs are supported."
            )

        parsed = urlparse(resolved_source) if source_kind == "url" else None
        filename = (
            PurePath(parsed.path).name if parsed and parsed.path else Path(resolved_source).name
        ) or "media"

        payload: dict[str, Any] = {
            "source": resolved_source,
            "source_type": source_kind,
            "filename": filename,
            "kind": file_type.kind,
            "mime_type": file_type.mime_type or "application/octet-stream",
            "size_bytes": len(data),
            "size_megabytes": round(len(data) / (1024 * 1024), 3),
        }

        if file_type.kind == "image":
            if dimensions := _extract_image_size(data):
                payload["width"], payload["height"] = dimensions
        elif file_type.kind == "audio":
            if audio_format := audio_format_from_mime(file_type.mime_type):
                payload["audio_format"] = audio_format

        if include_data_url and len(data) <= INLINE_DATA_URL_BYTES:
            payload["data_url"] = _to_data_url(payload["mime_type"], data)
        elif include_data_url:
            logger.debug(
                "read_media: data_url omitted (size {} > {} MB limit)",
                len(data),
                INLINE_DATA_URL_MEGABYTES,
            )
            payload["note"] = (
                f"Data URL omitted because file exceeds {INLINE_DATA_URL_MEGABYTES} MB inline limit."
            )

        logger.info(
            "read_media: success kind={} size={} bytes source_type={}",
            file_type.kind,
            len(data),
            source_kind,
        )
        return json.dumps(payload, ensure_ascii=True, indent=2)

    async def execute(self, path: str, include_data_url: bool = False, **kwargs: Any) -> str:
        try:
            return await self._read_media_file(path, include_data_url=include_data_url)
        except PermissionError as e:
            logger.error("read_media: permission denied path='{}': {}", path, e)
            return f"Error: {e}"
        except FileNotFoundError as e:
            logger.error("read_media: file not found path='{}': {}", path, e)
            return f"Error: {e}"
        except IsADirectoryError as e:
            logger.error("read_media: path is directory path='{}': {}", path, e)
            return f"Error: {e}"
        except httpx.HTTPError as e:
            logger.error("read_media: HTTP error path='{}': {}", path, e)
            return f"Error reading remote media: {e}"
        except Exception as e:
            logger.exception("read_media: unexpected error path='{}': {}", path, e)
            return f"Error reading media: {str(e)}"
