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

# --- 4b. Which harness(es) will this workspace use? -------------------------------------
echo
echo "Which AI harness(es) will use docusearch in this workspace? (space-separated for several)"
echo "  1) Claude Code"
echo "  2) Codex CLI"
echo "  3) Copilot CLI / VS Code Copilot"
printf "Select [1 2 3]: "; read -r harnesses
harnesses="${harnesses:-1 2 3}"
H_CLAUDE=0; H_CODEX=0; H_COPILOT=0
case " $harnesses " in *" 1 "*) H_CLAUDE=1 ;; esac
case " $harnesses " in *" 2 "*) H_CODEX=1 ;; esac
case " $harnesses " in *" 3 "*) H_COPILOT=1 ;; esac

# --- 5. Write the per-harness client configs --------------------------------------------
if [ "$H_CLAUDE" -eq 1 ]; then
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
  # Pre-approve the docusearch tools so headless (-p) runs never stall on a prompt.
  # NOTE: these entries only take effect once the workspace itself is trusted — run
  # `claude` here interactively once and accept the trust dialog.
  mkdir -p .claude
  if [ ! -f .claude/settings.json ]; then
    cat > .claude/settings.json <<'JSON'
{
  "enabledMcpjsonServers": ["docusearch"],
  "permissions": {
    "allow": [
      "mcp__docusearch__search_docs",
      "mcp__docusearch__get_document",
      "mcp__docusearch__related_documents",
      "mcp__docusearch__catalog_stats",
      "mcp__docusearch__report_format",
      "mcp__docusearch__build_report",
      "mcp__docusearch__verify_citations",
      "mcp__docusearch__get_image",
      "mcp__docusearch__describe_image",
      "mcp__docusearch__help",
      "mcp__docusearch__list_stores"
    ]
  }
}
JSON
    echo "✓ Claude Code: wrote .mcp.json + .claude/settings.json (tool pre-approval)"
    echo "  ! One-time step: run 'claude' here interactively and accept the trust dialog,"
    echo "    or the permissions.allow entries are ignored in headless mode."
  else
    echo "✓ Claude Code: wrote .mcp.json  (.claude/settings.json exists — merge the docusearch"
    echo "  enabledMcpjsonServers + permissions.allow entries yourself; see README)"
  fi
fi

if [ "$H_COPILOT" -eq 1 ]; then
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
  echo "✓ Copilot: wrote .vscode/mcp.json (VS Code picks it up on reload)"
  echo "  CLI invocation: copilot --additional-mcp-config '{\"mcpServers\":{\"docusearch\":{\"type\":\"http\",\"url\":\"${URL}\"}}}' --allow-tool 'docusearch(*)'"
fi

if [ "$H_CODEX" -eq 1 ]; then
  CODEX_BLOCK="
# --- added by docusearch install.sh ($(date +%Y-%m-%d)) ---
# workspace-write + network_access are REQUIRED: codex's sandbox otherwise blocks
# even localhost connections, and headless exec auto-cancels gated MCP calls.
sandbox_mode = \"workspace-write\"
approval_policy = \"never\"

[sandbox_workspace_write]
network_access = true

[mcp_servers.docusearch]
url = \"${URL%/mcp}/mcp\"
enabled = true
"
  if [ -f "$HOME/.codex/config.toml" ] && ! grep -q 'mcp_servers.docusearch' "$HOME/.codex/config.toml"; then
    echo
    echo "Codex CLI needs these lines in ~/.codex/config.toml (global, affects all Codex sessions):"
    echo "$CODEX_BLOCK"
    printf "Append them now? (y/N) "; read -r codexok
    if [ "${codexok:-n}" = "y" ] || [ "${codexok:-n}" = "Y" ]; then
      printf '%s\n' "$CODEX_BLOCK" >> "$HOME/.codex/config.toml"
      echo "✓ Codex: appended docusearch block to ~/.codex/config.toml"
    else
      echo "  Skipped — add the block yourself before using docusearch from Codex."
    fi
  else
    echo "✓ Codex: ~/.codex/config.toml already references docusearch (or does not exist — add the block after installing Codex)"
  fi
fi

echo "✓ MCP endpoint for this workspace: ${URL}"

# --- 5b. Preferred report output format (written into CLAUDE.md) -----------------------
echo
echo "What format should reports be produced in by default?"
echo "  1) HTML        — one standalone .html file (default)"
echo "  2) HTML slides — a browsable deck"
echo "  3) Markdown    — one self-contained .md file"
echo "  4) DOCX        — Word document"
echo "  5) XLSX        — Excel workbook"
echo "  6) PPTX        — PowerPoint deck"
echo "  7) PDF         — PDF document"
echo "  8) Inline chat — answer in the conversation, write no file"
printf "Select 1-8 [1]: "; read -r fmt

# 1-7 are real render formats and belong in docusearch.yaml under reports.default_format —
# that is the key the report_format tool reads. 8 (inline) is not a render format, so it is
# expressed as an instruction in CLAUDE.md instead.
DEFAULT_FMT=""; INLINE=0
case "${fmt:-1}" in
  2) DEFAULT_FMT="html-slide" ;;
  3) DEFAULT_FMT="md"   ;;
  4) DEFAULT_FMT="docx" ;;
  5) DEFAULT_FMT="xlsx" ;;
  6) DEFAULT_FMT="pptx" ;;
  7) DEFAULT_FMT="pdf"  ;;
  8) INLINE=1 ;;
  *) DEFAULT_FMT="html" ;;
esac

if [ -n "$DEFAULT_FMT" ] && [ -f docusearch.yaml ]; then
  if grep -qE '^\s*default_format:' docusearch.yaml; then
    DEFAULT_FMT="$DEFAULT_FMT" perl -i -pe 's/^(\s*default_format:\s*).*$/$1"$ENV{DEFAULT_FMT}"/ if !$done && /^\s*default_format:/ and ($done=1)' docusearch.yaml
    echo "✓ reports.default_format = \"$DEFAULT_FMT\"  (docusearch.yaml)"
  else
    echo "! docusearch.yaml has no reports.default_format key — set it by hand." >&2
  fi
fi

# --- 5b2. Response & report style ---------------------------------------------------------
echo
echo "Preferred response/report verbosity for this workspace?"
echo "  1) concise  — lead with the answer; tables over prose; minimal background"
echo "  2) standard — balanced (default)"
echo "  3) detailed — full explanations, background, and caveats spelled out"
printf "Select 1-3 [2]: "; read -r vsel
case "${vsel:-2}" in
  1) STYLE_LINE="Verbosity: CONCISE. Lead with the answer; prefer tables to prose; omit background the reader did not ask for." ;;
  3) STYLE_LINE="Verbosity: DETAILED. Spell out reasoning, background, and caveats in full prose." ;;
  *) STYLE_LINE="Verbosity: STANDARD. Balance brevity with completeness." ;;
esac
STYLE_LINE="$STYLE_LINE
Style is personal: if the user asks about style or formatting options, present the menu —
report themes (midnight / paper / slate / contrast), verbosity (concise / standard /
detailed), and output formats (md / html / html-slide / pdf / docx / pptx / xlsx) — and
apply their choice for that report or, if they say so, from then on."

# --- 5c. Report tooling choice (policy: html always docusearch; Copilot always docusearch) ---
# For pptx/docx/xlsx/pdf the operator may prefer the harness's own document tooling.
TOOLING="docusearch"
case "$DEFAULT_FMT" in
  docx|xlsx|pptx|pdf)
    if [ "$H_COPILOT" -eq 1 ] && [ "$H_CLAUDE" -eq 0 ] && [ "$H_CODEX" -eq 0 ]; then
      echo "Report tooling: docusearch build_report (Copilot CLI has no native renderer — policy)"
    else
      echo
      echo "Who should render $DEFAULT_FMT reports by default?"
      echo "  1) docusearch build_report — citations machine-verified, file hosted on the server (default)"
      echo "  2) harness-native tooling  — the AI renders the document itself; it must then pass"
      echo "     verify_citations before delivering (Copilot CLI always uses docusearch regardless)"
      printf "Select 1/2 [1]: "; read -r tsel
      [ "${tsel:-1}" = "2" ] && TOOLING="native"
    fi
    ;;
  *) : ;;  # html/html-slide/md: always docusearch build_report (policy)
esac

if [ "$INLINE" -eq 1 ]; then
  FMT_BLOCK='**Answer inline.** Reply in the conversation as well-structured Markdown, with
every citation resolved to a real document. This mode writes **no** report file — only call
`build_report` if the user explicitly asks for a file.'
elif [ "$TOOLING" = "native" ]; then
  FMT_BLOCK="**Reports in ${DEFAULT_FMT} are rendered by YOUR OWN document tooling** (operator's
choice) — but grounding rules do not relax:

1. Research through search_docs / get_document as usual; cite every catalog fact
   [D:doc#chunk], general knowledge [GK].
2. Render the ${DEFAULT_FMT} with your native tooling.
3. **Before delivering, call verify_citations(text, evidence=[[doc_id, chunk_id], ...])**
   with the full body text and the pairs you actually retrieved. Do not ship while ok=false.
   Use its 'resolved' href+label pairs for the references section.
4. html / html-slide / md deliverables are the exception: those ALWAYS go through
   build_report. On the Copilot CLI harness, ALL formats go through build_report."
else
  FMT_BLOCK='**Reports are files, and the server writes them — you deliver the link.**

1. Call `report_format()` **before you draft.** It returns the operator'"'"'s configured
   default (`reports.default_format` in `docusearch.yaml`) and how to author for that
   target — a deck needs short bullets, a spreadsheet one fact per row, a document prose.
2. **If the requester names a format, that wins**; the configured default applies only when
   they did not say.
3. Call `build_report(spec, fmt=...)`. It verifies every citation, saves the file under
   `tmp_dir/reports/`, and returns `{fmt, filename, url, bytes}` — **give the user the
   `url`.** Do not re-write the file yourself and never hand-format a report.

One report, one file.'
fi

# DOCX/XLSX/PPTX/PDF are rendered by the [export] extra. PDF additionally renders the real
# HTML through headless chromium, so Playwright needs its browser downloaded once.
case "${fmt:-1}" in
  4|5|6|7)
    echo
    echo "This format needs the export renderers — installing docusearch[export] …"
    pip install -e ".[export]" -q && echo "✓ export renderers installed"
    if [ "${fmt}" = "7" ]; then
      echo "PDF renders through headless chromium; downloading it once (~150 MB) …"
      if python -m playwright install chromium; then
        echo "✓ chromium ready — PDF export will work"
      else
        echo "! chromium download failed. PDF export will report an export-dependency" >&2
        echo "  error until you run:  python -m playwright install chromium" >&2
      fi
    fi ;;
esac

FMT_BLOCK="$FMT_BLOCK

$STYLE_LINE"

if [ -f CLAUDE.md ] && grep -q 'docusearch:output-format:start' CLAUDE.md; then
  FMT_BLOCK="$FMT_BLOCK" perl -0777 -i -pe '
    my $b = $ENV{FMT_BLOCK};
    s{(<!-- docusearch:output-format:start -->\n).*?(<!-- docusearch:output-format:end -->)}
     {$1$b\n$2}s;
  ' CLAUDE.md
  echo "✓ Report output format written into CLAUDE.md"
else
  echo "! Could not find the output-format markers in CLAUDE.md — left unchanged." >&2
fi

# Codex and the Copilot CLI read AGENTS.md, not CLAUDE.md — keep them identical so no
# harness gets different instructions (they must not diverge).
if [ "$H_CODEX" -eq 1 ] || [ "$H_COPILOT" -eq 1 ]; then
  if [ -f CLAUDE.md ]; then
    cp CLAUDE.md AGENTS.md
    echo "✓ AGENTS.md written (identical copy of CLAUDE.md for Codex / Copilot CLI)"
  fi
fi

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
