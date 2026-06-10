#!/usr/bin/env bash
# One-click launcher: starts the Odysseus NVIDIA GPU stack and opens the browser.
#
# Install:
#   chmod +x /media/dari/Above-Average/odysseus/scripts/launch-odysseus.sh
#   cp scripts/odysseus.desktop.template ~/.local/share/applications/odysseus.desktop
#   update-desktop-database ~/.local/share/applications/
#
# Logs written to /tmp/odysseus-launch.log for easy debugging.

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${ODYSSEUS_DIR:-/media/dari/Above-Average/odysseus}"
PORT="${APP_PORT:-7000}"
COMPOSE_FILE="docker-compose.gpu-nvidia.yml"
LOG="/tmp/odysseus-launch.log"
MAX_WAIT=120  # seconds
# ──────────────────────────────────────────────────────────────────────────────

exec > >(tee -a "$LOG") 2>&1
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Odysseus launcher starting"

# ── Dependency checks ─────────────────────────────────────────────────────────
die() { echo "[ERROR] $*" >&2; notify-send "Odysseus" "Error: $*" 2>/dev/null || true; exit 1; }

command -v docker   >/dev/null 2>&1 || die "docker not found — install Docker Desktop or Engine"
command -v curl     >/dev/null 2>&1 || die "curl not found — install curl"

# Prefer 'docker compose' (v2); fall back to 'docker-compose' (v1)
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  die "Neither 'docker compose' nor 'docker-compose' found"
fi

# ── Project checks ────────────────────────────────────────────────────────────
[ -d "$PROJECT" ] || die "Project directory not found: $PROJECT (set ODYSSEUS_DIR to override)"
COMPOSE_PATH="$PROJECT/$COMPOSE_FILE"
[ -f "$COMPOSE_PATH" ] || die "Compose file not found: $COMPOSE_PATH"

cd "$PROJECT"

# ── Check if already running ──────────────────────────────────────────────────
if curl -sf "http://localhost:${PORT}" > /dev/null 2>&1; then
  echo "[$(date '+%H:%M:%S')] Odysseus already running on port $PORT — opening browser"
  xdg-open "http://localhost:${PORT}"
  exit 0
fi

# ── Start stack ───────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Starting $COMPOSE_FILE ..."
if ! $COMPOSE -f "$COMPOSE_FILE" up -d 2>&1; then
  die "docker compose up failed — check $LOG for details"
fi

# ── Wait for app ──────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for Odysseus on port $PORT (up to ${MAX_WAIT}s)..."
START=$(date +%s)
while true; do
  if curl -sf "http://localhost:${PORT}" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Odysseus is up ✓"
    break
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[$(date '+%H:%M:%S')] Timed out after ${MAX_WAIT}s waiting for port $PORT"
    echo "Check container logs: docker compose -f $COMPOSE_FILE logs --tail 50"
    notify-send "Odysseus" "Startup timed out — see $LOG" 2>/dev/null || true
    # Still try to open (app may be slow but listening)
    xdg-open "http://localhost:${PORT}"
    exit 1
  fi
  sleep 2
done

# ── Open browser ──────────────────────────────────────────────────────────────
xdg-open "http://localhost:${PORT}"
echo "[$(date '+%H:%M:%S')] Done"
