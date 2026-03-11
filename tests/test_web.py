import json

import httpx
import pytest

from winclaw.tools.web import WebFetchTool, WebSearchTool


class MockTavilyClient:
    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "title": "Result One",
                    "url": "https://example.com/1",
                    "content": "First result",
                }
            ]
        }


class MockAsyncClient:
    def __init__(self, response: httpx.Response, **kwargs):
        self.response = response
        self.kwargs = kwargs
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_web_fetch_rejects_invalid_url():
    tool = WebFetchTool()

    result = json.loads(await tool.execute("file:///tmp/test.html"))

    assert result["url"] == "file:///tmp/test.html"
    assert "URL validation failed" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_converts_html_to_markdown(monkeypatch):
    html = """
    <html>
      <head>
        <title>Example Page</title>
        <style>.hidden { display:none; }</style>
      </head>
      <body>
        <script>console.log("ignore me")</script>
        <h1>Hello</h1>
        <p>Visit <a href="https://example.com/docs">the docs</a>.</p>
      </body>
    </html>
    """
    response = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=html,
        request=httpx.Request("GET", "https://example.com/page"),
    )
    client = MockAsyncClient(response, follow_redirects=True)
    monkeypatch.setattr("winclaw.tools.web.httpx.AsyncClient", lambda **kwargs: client)

    result = json.loads(await WebFetchTool().execute("https://example.com/page"))

    assert result["status"] == 200
    assert result["finalUrl"] == "https://example.com/page"
    assert result["extractor"] == "markdownify"
    assert result["truncated"] is False
    assert result["text"].startswith("# Example Page")
    assert "# Hello" in result["text"]
    assert "[the docs](https://example.com/docs)" in result["text"]
    assert "ignore me" not in result["text"]


@pytest.mark.asyncio
async def test_web_fetch_text_mode_supports_legacy_kwargs(monkeypatch):
    html = """
    <html>
      <head><title>Example Page</title></head>
      <body><p>Hello <strong>world</strong> and friends.</p></body>
    </html>
    """
    response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text=html,
        request=httpx.Request("GET", "https://example.com/text"),
    )
    monkeypatch.setattr(
        "winclaw.tools.web.httpx.AsyncClient",
        lambda **kwargs: MockAsyncClient(response, **kwargs),
    )

    result = json.loads(
        await WebFetchTool().execute(
            "https://example.com/text",
            extractMode="text",
            maxChars=20,
        )
    )

    assert result["extractor"] == "markdownify"
    assert result["truncated"] is True
    assert result["length"] == 20
    assert result["text"].startswith("# Example Page\n\n")


@pytest.mark.asyncio
async def test_web_search_formats_results_and_clamps_count(monkeypatch):
    client = MockTavilyClient()
    monkeypatch.setattr("winclaw.tools.web.TavilyClient", lambda **kwargs: client)

    result = await WebSearchTool(api_key="token").execute("cursor", count=20)

    assert "Results for: cursor" in result
    assert "1. Result One" in result
    assert "https://example.com/1" in result
    assert "First result" in result
    assert client.api_key == "token"
    assert client.calls[0] == {
        "query": "cursor",
        "search_depth": "advanced",
        "max_results": 10,
        "include_answer": False,
        "include_raw_content": False,
    }
