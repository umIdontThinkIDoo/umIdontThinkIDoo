#!/usr/bin/env bash
# One-click launcher: starts the Odysseus NVIDIA GPU stack and opens the browser.
# Install: chmod +x scripts/launch-odysseus.sh
# Desktop: copy scripts/odysseus.desktop.template to ~/.local/share/applications/odysseus.desktop
set -euo pipefail

PROJECT="/media/dari/Above-Average/odysseus"
PORT="${APP_PORT:-7000}"

cd "$PROJECT"
docker compose -f docker-compose.gpu-nvidia.yml up -d

echo "Waiting for Odysseus on port $PORT..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}" > /dev/null 2>&1; then
    break
  fi
  sleep 2
done

xdg-open "http://localhost:${PORT}"
