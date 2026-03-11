"""Web tools: web_search and web_fetch."""

import asyncio
import html
import json
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger
from markdownify import markdownify as md
from tavily import TavilyClient

from winclaw.config.loader import get_config_path
from winclaw.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_html(html_content: str) -> str:
    """Remove non-content tags before extraction."""
    html_content = re.sub(r"<script[\s\S]*?</script>", "", html_content, flags=re.I)
    return re.sub(r"<style[\s\S]*?</style>", "", html_content, flags=re.I)


def _extract_title(html_content: str) -> str:
    """Extract document title from HTML."""
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_content, flags=re.I)
    return _normalize(_strip_tags(match.group(1))) if match else ""


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using Tavily Search API."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results of web search "},
        },
        "required": ["query"],
    }

    def __init__(
        self, api_key: Optional[str] = None, max_results: int = 5, proxy: Optional[str] = None
    ):
        self._init_api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("TAVILY_API_KEY", "")

    async def execute(self, query: str, count: Optional[int] = None, **kwargs: Any) -> str:
        if not self.api_key:
            return (
                f"Error: Tavily API key not configured. Set it in "
                f"{get_config_path()} under tools.web.search.apiKey "
                "(or export TAVILY_API_KEY), then restart the gateway."
            )

        try:
            n = min(max(count or self.max_results, 1), 5)
            if self.proxy:
                logger.debug("WebSearch: Tavily client ignores tool-level proxy setting")
            client = TavilyClient(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.search,
                query=query,
                search_depth="advanced",
                max_results=n,
                include_answer=False,
                include_raw_content=False,
            )

            results = response.get("results", [])[:n]
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("content"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except httpx.ProxyError as e:
            logger.error("WebSearch proxy error: {}", e)
            return f"Proxy error: {e}"
        except Exception as e:
            logger.error("WebSearch error: {}", e)
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using markdownify."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = 50000, proxy: Optional[str] = None):
        self.max_chars = max_chars
        self.proxy = proxy

    async def execute(
        self,
        url: str,
        extract_mode: str = "markdown",
        max_chars_arg: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        extract_mode = kwargs.get("extractMode", extract_mode)
        max_chars_arg = kwargs.get("maxChars", max_chars_arg)
        max_chars = max_chars_arg or self.max_chars
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps(
                {"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False
            )

        try:
            logger.debug("WebFetch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
                proxy=self.proxy,
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                clean_html = _clean_html(r.text)
                content = (
                    _normalize(md(clean_html, heading_style="ATX"))
                    if extract_mode == "markdown"
                    else _normalize(_strip_tags(clean_html))
                )
                title = _extract_title(r.text)
                text = f"# {title}\n\n{content}" if title and content else content or f"# {title}"
                extractor = "markdownify"
            else:
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(r.url),
                    "status": r.status_code,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except httpx.ProxyError as e:
            logger.error("WebFetch proxy error for {}: {}", url, e)
            return json.dumps({"error": f"Proxy error: {e}", "url": url}, ensure_ascii=False)
        except Exception as e:
            logger.error("WebFetch error for {}: {}", url, e)
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)
