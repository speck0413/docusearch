#!/usr/bin/env bash
# start-server.sh — start (or restart) the docusearch MCP + REST server.
#
# Meant for a DEDICATED SERVER: it binds 0.0.0.0 by default so clients on other
# machines can connect (http://<this-host>:8321/mcp). Run it on the box that holds the
# catalog; laptops/clients just point their MCP config at this server's URL.
#
# Running this script starts the server in the background off the project's .venv. If a
# server is already running it asks whether to restart it (y/N).
#
# Env overrides:  DOCUSEARCH_CONFIG (default ./docusearch.yaml)
#                 DOCUSEARCH_HOST   (default 0.0.0.0)
#                 DOCUSEARCH_PORT   (default 8321)
#
# For an always-on production service, prefer a supervisor (systemd/NSSM) — see
# SERVER-SETUP-GUIDE.md. This script is the quick manual launcher.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

VENV="$here/.venv"
CONFIG="${DOCUSEARCH_CONFIG:-$here/docusearch.yaml}"
HOST="${DOCUSEARCH_HOST:-0.0.0.0}"
PORT="${DOCUSEARCH_PORT:-8321}"
PIDFILE="$here/.docusearch-server.pid"
LOGFILE="$here/tmp/server.log"

if [ ! -x "$VENV/bin/docusearch" ]; then
  echo "! docusearch is not installed in $VENV — run ./install.sh first." >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "! No config at $CONFIG — run ./install.sh (or 'docusearch init') first." >&2
  exit 1
fi

is_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

port_busy() {
  "$VENV/bin/python" - "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(0.4)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

stop_running() {
  local pid; pid="$(cat "$PIDFILE")"
  echo "Stopping docusearch server (pid $pid)…"
  kill "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do is_running || break; sleep 0.5; done
  if is_running; then kill -9 "$pid" 2>/dev/null || true; sleep 0.5; fi
  rm -f "$PIDFILE"
}

# Already running under this script? Offer to restart.
if is_running; then
  printf "docusearch server is already running (pid %s). Restart it? (y/N) " "$(cat "$PIDFILE")"
  read -r ans
  case "${ans:-N}" in
    y|Y|yes|YES) stop_running ;;
    *) echo "Leaving the running server as-is. (URL: http://$HOST:$PORT/mcp)"; exit 0 ;;
  esac
# Running but not started by us (no/stale pidfile, port taken)? Don't stomp it.
elif port_busy; then
  echo "! Port $PORT is already in use by another process (not started by this script)." >&2
  echo "  Free it, stop that process, or set DOCUSEARCH_PORT to a different port." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOGFILE")"
echo "Starting docusearch server on $HOST:$PORT  (config: $CONFIG)…"
nohup "$VENV/bin/docusearch" serve --config "$CONFIG" --host "$HOST" --port "$PORT" \
  >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

# Give uvicorn a moment, then confirm it's actually up.
for _ in 1 2 3 4 5 6 7 8 9 10; do is_running && port_busy && break; sleep 0.5; done
if is_running && port_busy; then
  echo "✓ Server up (pid $(cat "$PIDFILE"))."
  echo "    MCP:  http://$HOST:$PORT/mcp"
  echo "    REST: http://$HOST:$PORT/docs"
  echo "    Logs: $LOGFILE   ·   Stop: kill \$(cat $PIDFILE)"
else
  echo "! Server failed to start — last lines of $LOGFILE:" >&2
  tail -n 20 "$LOGFILE" >&2 2>/dev/null || true
  rm -f "$PIDFILE"
  exit 1
fi
