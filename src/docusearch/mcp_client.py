"""A tiny **synchronous** MCP client over streamable HTTP (§10, R-API-1).

This is the transport behind the ``docusearch stdf`` CLI tools: they drive the *same* MCP server an
AI agent connects to, so "does it work over the wire" is proven every time the CLI is used — the
standing CLI↔MCP feature-parity check, not a second in-process code path.

The client is deliberately thin: one call opens a session, invokes a tool, and closes. That is a few
extra round-trips per command, which is irrelevant for an interactive CLI and keeps the surface
trivial (no connection pool, no background loop). Errors collapse to one actionable line — an
unreachable server tells you to start ``docusearch serve`` (operability, not a traceback).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class MCPError(RuntimeError):
    """A failure talking to the MCP server (unreachable, protocol error, or a tool-level error)."""


def _flatten(exc: BaseException) -> list[BaseException]:
    """Flatten a (possibly nested) ExceptionGroup — anyio/asyncio wrap transport failures in one."""
    group = getattr(exc, "exceptions", None)
    if group is None:
        return [exc]
    out: list[BaseException] = []
    for sub in group:
        out.extend(_flatten(sub))
    return out


def _friendly(url: str, exc: BaseException) -> str:
    """Turn a transport exception into one actionable line."""
    leaves = _flatten(exc)
    text = "; ".join(f"{type(e).__name__}: {e}" for e in leaves if str(e)) or type(exc).__name__
    lowered = text.lower()
    if any(s in lowered for s in ("connect", "refused", "connection", "timeout", "unreachable")):
        return (
            f"cannot reach the docusearch MCP server at {url} ({text}). "
            "Start it with `docusearch serve`, or pass --url http://HOST:PORT/mcp."
        )
    return f"MCP call to {url} failed: {text}"


class MCPClient:
    """Call MCP tools on a running ``docusearch serve`` by URL (e.g. ``http://127.0.0.1:8321/mcp``)."""

    def __init__(self, url: str, *, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout

    def call(self, tool: str, **arguments: Any) -> Any:
        """Invoke ``tool`` with keyword ``arguments``; return its structured result (usually a dict).
        Raises :class:`MCPError` on an unreachable server, a protocol error, or a tool-reported
        error (a result flagged ``isError``)."""
        import gc
        import warnings

        try:
            return asyncio.run(self._call(tool, arguments))
        finally:
            # asyncio.run closes the loop but the streamable-HTTP transport leaves anyio memory
            # streams for the collector; collect them now under a ResourceWarning filter so their
            # __del__ can't surface a stray warning (which -W error would escalate to an error).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        try:
            # can't merge these: ClientSession consumes the streams the first context unpacks.
            async with streamablehttp_client(self.url) as (read, write, _get_sid):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, arguments)
        except MCPError:
            raise
        except BaseException as exc:  # noqa: BLE001 - collapse any transport failure to one line
            raise MCPError(_friendly(self.url, exc)) from exc
        return _unwrap(result, tool)


def _unwrap(result: Any, tool: str) -> Any:
    """Pull the payload out of a CallToolResult: prefer structured content, fall back to text JSON."""
    if getattr(result, "isError", False):
        raise MCPError(f"tool {tool!r} returned an error: {_result_text(result)}")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps a non-object return (str/list) as {"result": ...}; unwrap that one case.
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    text = _result_text(result)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _result_text(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)
