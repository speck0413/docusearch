# docusearch — Server Setup Guide

`GETTING_STARTED.md` runs docusearch **locally** — one person, everything on their own
machine, the MCP server bound to `localhost`. This guide covers the other deployment: a
**dedicated server** that ingests the corpus once and serves a **central MCP + REST
endpoint** the whole team connects to.

Same binary, same commands — the difference is entirely in **configuration**: where the
data lives, what address the server binds, and how clients find it.

## Local vs. server at a glance

| | Local (laptop) | Dedicated server |
|---|---|---|
| `mode` | `standalone` | `server` (heavy host) — clients use `client` |
| `serve.host` | `127.0.0.1` (localhost only) | `0.0.0.0` (reachable on the network) |
| `paths.*` | relative (`./catalog.db`) | **absolute**, on fast local disk (`/srv/docusearch/...`) |
| `embed.model` | small (`all-MiniLM-L6-v2`) | larger is fine (`bge-large-en-v1.5`) — the server has the RAM/CPU |
| Who ingests | the user | an admin, on the server |
| Client MCP URL | `http://localhost:8321/mcp` | `http://<server-host>:8321/mcp` |
| Lifecycle | run `serve` by hand | run `serve` as a managed **service** |

Everything stays **local to your network** — no document content leaves the server.

## 1. Install on the server

```bash
# bash
python3 -m venv /srv/docusearch/.venv
source /srv/docusearch/.venv/bin/activate
pip install -e ".[embeddings,server]"     # vector search + REST/MCP
docusearch --version
```

```powershell
# PowerShell
python -m venv C:\docusearch\.venv
C:\docusearch\.venv\Scripts\Activate.ps1
pip install -e ".[embeddings,server]"
docusearch --version
```

## 2. Write the server config

`docusearch init` writes a commented `docusearch.yaml`; edit these blocks for a server.
The key differences from a local config are **absolute paths**, **`host: 0.0.0.0`**, and
(optionally) a **larger model**:

```yaml
mode: "server"                     # this box does the ingestion + indexing + serving

paths:
  staging_dir: "/srv/docusearch/staging"   # absolute; fast local disk, not a network share
  db_path: "/srv/docusearch/catalog.db"
  tmp_dir: "/srv/docusearch/tmp"

sources:
  - type: "fs"
    name: "vendor-html"
    location: "/srv/docs/vendor-html"       # where the corpus lives ON THE SERVER
    include: ["**/*.html"]
    content_selector: ""                     # tighten once you see the audit
    audience: ["engineering"]

embed:
  model: "BAAI/bge-large-en-v1.5"            # a server can afford the best model
  device: "auto"                             # CUDA GPU if present, else CPU

serve:
  host: "0.0.0.0"                            # bind on all interfaces so clients can reach it
  port: 8321
```

Windows paths work too (`D:\docusearch\catalog.db`, `location: "D:/docs/vendor-html"`).

> The **model cache** (`~/.cache/huggingface/hub`, or `HF_HOME`) lives on the server. Download
> the model once while online; after that the server runs fully offline.

## 3. Ingest on the server

```bash
docusearch ingest --config /srv/docusearch/docusearch.yaml     # parses across all cores
docusearch audit  --config /srv/docusearch/docusearch.yaml     # loud list of anything skipped
```

Re-run `ingest` on a schedule (cron / Task Scheduler) to keep the catalog fresh — it's
incremental, so only changed files (by SHA-256) are re-processed.

## 4. Run `serve` as a managed service

Foreground (to smoke-test): `docusearch serve --config /srv/docusearch/docusearch.yaml`.
For production, keep it running under a supervisor so it restarts on crash/reboot.

**Linux — systemd** (`/etc/systemd/system/docusearch.service`):

```ini
[Unit]
Description=docusearch MCP + REST server
After=network.target

[Service]
ExecStart=/srv/docusearch/.venv/bin/docusearch serve --config /srv/docusearch/docusearch.yaml
Environment=HF_HOME=/srv/docusearch/models
Restart=on-failure
User=docusearch

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now docusearch
sudo systemctl status docusearch
```

**Windows** — run it as a service with [NSSM](https://nssm.cc/) (or a Task Scheduler task
"At startup"):

```powershell
nssm install docusearch C:\docusearch\.venv\Scripts\docusearch.exe `
  serve --config C:\docusearch\docusearch.yaml
nssm start docusearch
```

## 5. Networking

- **Firewall:** open the `serve.port` (default **8321/tcp**) to your LAN only.
- **Hostname:** give the box a stable name (`docs-server.local`) so client configs don't
  chase an IP.
- **TLS (recommended off-LAN):** docusearch serves plain HTTP. To expose it beyond a
  trusted network, front it with a reverse proxy (nginx/Caddy) terminating HTTPS and
  proxying to `127.0.0.1:8321`; clients then use `https://docs-server.example.com/mcp`.

## 6. Point clients at the server

Clients need **no local index** — just the server's URL. Two options:

**a) MCP/REST clients (Claude Code, Claude Desktop, VS Code/Copilot):** register the
server's MCP URL instead of localhost:

```bash
claude mcp add --transport http docusearch http://docs-server.local:8321/mcp
```

or in `.mcp.json` / `.vscode/mcp.json`, replace `localhost` with the server host:

```jsonc
{ "mcpServers": {                              // .vscode/mcp.json uses "servers"
    "docusearch": { "type": "http", "url": "http://docs-server.local:8321/mcp" }
} }
```

**b) The `docusearch` CLI as a thin client** (no local data, negotiates embeddings with the
server):

```yaml
mode: "client"
server_url: "http://docs-server.local:8321"
embed:
  model: "auto"          # use the server's model if it fits locally, else send plain text
```

```bash
docusearch search --config client.yaml "match loop single bit"
```

## 7. Permissions across the team

Audience scoping is per-client: set `DOCUSEARCH_ROLES` in the environment of each client (or
the MCP-server process that owns a user's session). A query only returns documents whose
`audience` intersects the caller's roles. See `GETTING_STARTED.md` §1.6.

```bash
export DOCUSEARCH_ROLES=engineering,test-eng     # bash
```
```powershell
$env:DOCUSEARCH_ROLES = "engineering,test-eng"   # PowerShell
```

Note (from §1.6): audience filtering is **cooperative** scoping for trusted internal users,
not a cryptographic boundary — use OS-level permissions on `paths.db_path` for hard limits.
