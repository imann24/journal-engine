#!/usr/bin/env bash
# Launch the journal web UI, bound to JOURNAL_WEB_HOST:JOURNAL_WEB_PORT.
# Reachable from the Mac over Tailscale at http://<spark-magicdns>:<port>.
# Keep it tailnet-only: do not port-forward or expose publicly.
set -euo pipefail
cd "$(dirname "$0")"

# Load .env so JOURNAL_PASSWORD / overrides are present (without overriding
# anything already exported).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${JOURNAL_WEB_HOST:-0.0.0.0}"
PORT="${JOURNAL_WEB_PORT:-8501}"

if [[ -z "${JOURNAL_PASSWORD:-}" ]]; then
  echo "ERROR: JOURNAL_PASSWORD is not set. Put it in .env (gitignored)." >&2
  exit 1
fi

# Prefer the project venv if present.
PY="python3"
[[ -x .venv/bin/python ]] && PY=".venv/bin/python"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

MAGICDNS="$(tailscale status --json 2>/dev/null \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)"

echo "Starting Journal Engine web UI on ${HOST}:${PORT}"
if [[ -n "$MAGICDNS" ]]; then
  echo "Open from your Mac (over Tailscale):  http://${MAGICDNS}:${PORT}"
fi

exec "$PY" -m streamlit run app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
