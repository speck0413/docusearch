"""Agent instruction files stay valid + carry the contract (§10, §11)."""

from __future__ import annotations

import json
from pathlib import Path

_AGENTS = Path(__file__).resolve().parents[1] / "agents"


def test_mcp_json_points_at_http_mcp() -> None:
    data = json.loads((_AGENTS / "mcp.json").read_text(encoding="utf-8"))
    server = data["mcpServers"]["docusearch"]
    assert server["url"].endswith("/mcp")  # streamable HTTP mount (not stdio)
    assert server["url"].startswith("http")


def test_claude_instructions_cover_citations_and_tools() -> None:
    text = (_AGENTS / "CLAUDE.md").read_text(encoding="utf-8")
    for needed in ("[GK]", "[D:", "search_docs", "get_document", "related_documents"):
        assert needed in text
    assert "must" in text.lower()  # citations are mandatory


def test_copilot_instructions_exist_and_cover_citations() -> None:
    text = (_AGENTS / "copilot-instructions.md").read_text(encoding="utf-8")
    assert "[GK]" in text and "[D:" in text and "search_docs" in text
