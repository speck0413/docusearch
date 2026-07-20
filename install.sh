#!/usr/bin/env bash
# install.sh — one-shot setup wizard for docusearch.
#
#   1. creates a Python virtual environment (.venv) and installs docusearch into it
#   2. writes a starter docusearch.yaml if you don't have one
#   3. asks whether the MCP server is LOCAL or REMOTE and writes the client MCP configs
#      (.mcp.json for Claude, .vscode/mcp.json for VS Code / Copilot) to match
#   4. prints how to connect Claude Code, Claude Desktop, and Copilot / VS Code
#
# The docusearch MCP server speaks HTTP, so it must be running before a client connects.
# For LOCAL use, start it with ./start-server.sh first. For REMOTE use, a dedicated server
# runs it (see SERVER-SETUP-GUIDE.md); this machine only needs the URL.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

echo "docusearch installer"
echo "===================="

# --- 1. Python + venv ------------------------------------------------------------------
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "! $PY not found. Install Python 3.11+ and re-run (or set PYTHON=...)." >&2
  exit 1
fi
if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv)…"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

# --- 2. Dependencies -------------------------------------------------------------------
echo
echo "Vector (hybrid) search downloads a model + PyTorch (bigger, better for conceptual"
echo "queries). Without it you get fast BM25 keyword search only (no model download)."
printf "Enable vector/hybrid search? (Y/n) "; read -r emb
case "${emb:-Y}" in
  n|N|no|NO) EXTRAS="server" ;;
  *)         EXTRAS="embeddings,server" ;;
esac
echo "Installing docusearch[$EXTRAS] …"
python -m pip install --upgrade pip -q
pip install -e ".[$EXTRAS]" -q
echo "✓ Installed $(docusearch --version)"

# --- 3. Config -------------------------------------------------------------------------
if [ ! -f docusearch.yaml ]; then
  echo
  echo "Writing a starter docusearch.yaml …"
  docusearch init >/dev/null
  echo "✓ docusearch.yaml created — edit sources[].location and embed.model before ingesting."
else
  echo "✓ Using existing docusearch.yaml"
fi

# --- 4. Local vs remote MCP ------------------------------------------------------------
echo
echo "Where does the docusearch SERVER run?"
echo "  1) local   — on this machine (you start it with ./start-server.sh)"
echo "  2) remote  — a dedicated server already running docusearch"
printf "Select 1/2 [1]: "; read -r mode

LOCAL=0
case "${mode:-1}" in
  2)
    printf "  Remote server host (e.g. docs-server.local): "; read -r rhost
    printf "  Port [8321]: "; read -r rport; rport="${rport:-8321}"
    URL="http://${rhost}:${rport}/mcp"
    ;;
  *)
    LOCAL=1
    URL="http://localhost:8321/mcp"
    ;;
esac

# --- 5. Write the two MCP client configs ----------------------------------------------
cat > .mcp.json <<JSON
{
  "mcpServers": {
    "docusearch": {
      "type": "http",
      "url": "${URL}"
    }
  }
}
JSON

mkdir -p .vscode
cat > .vscode/mcp.json <<JSON
{
  "servers": {
    "docusearch": {
      "type": "http",
      "url": "${URL}"
    }
  }
}
JSON

echo "✓ Wrote .mcp.json and .vscode/mcp.json  ->  ${URL}"

# --- 6. What to do next ----------------------------------------------------------------
cat <<EOF

Setup complete.
===============
EOF

if [ "$LOCAL" -eq 1 ]; then
  cat <<EOF
This is a LOCAL setup. The MCP server must be running before you open a client:

    1. Ingest your docs (first time):   docusearch ingest && docusearch audit
    2. START THE SERVER:                ./start-server.sh
    3. THEN launch Claude Code / VS Code (they connect to http://localhost:8321/mcp)

Leave the server running while you work; ./start-server.sh will offer to restart it.
EOF
else
  cat <<EOF
This is a REMOTE setup. Clients here connect to ${URL}.
The dedicated server must be running docusearch (see SERVER-SETUP-GUIDE.md, ./start-server.sh).
EOF
fi

cat <<EOF

Connect your client to "docusearch":

  • Claude Code (CLI):
      claude mcp add --transport http docusearch ${URL}
      claude mcp list                      # verify

  • Claude Desktop:
      Settings -> Developer -> Edit Config (claude_desktop_config.json), add:
        macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
        Windows: %APPDATA%\\Claude\\claude_desktop_config.json
      {
        "mcpServers": { "docusearch": { "type": "http", "url": "${URL}" } }
      }
      Restart Claude Desktop.

  • VS Code / GitHub Copilot:
      .vscode/mcp.json (written above) is picked up automatically — reload the window,
      then enable "docusearch" in the Copilot/Chat MCP servers list. The generic prompt
      in CLAUDE.md and the .claude/skills/ folder are read by both Claude and Copilot.

See GETTING_STARTED.md for the full walkthrough, SERVER-SETUP-GUIDE.md for a team server.
EOF
