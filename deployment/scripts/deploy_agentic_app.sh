#!/bin/bash

# Deploy Benjamin Agentic App service and timer
set -e

PROJECT_DIR="/home/chorizo/projects/benjamin_ai_system"
VENV_PATH="/home/chorizo/bbox_env"
SYSTEMD_DIR="/etc/systemd/system"

echo "[INFO] Installing agentic discussion oneshot service and timer..."
cp "$PROJECT_DIR/deployment/systemd/benjamin-agentic-discussion.service" "$SYSTEMD_DIR/"
cp "$PROJECT_DIR/deployment/systemd/benjamin-agentic-discussion.timer" "$SYSTEMD_DIR/"
chmod 644 "$SYSTEMD_DIR/benjamin-agentic-discussion.service" "$SYSTEMD_DIR/benjamin-agentic-discussion.timer"

echo "[INFO] Disabling deprecated telegram scheduler if present..."
systemctl disable --now benjamin-telegram-scheduler.service 2>/dev/null || true
rm -f "$SYSTEMD_DIR/benjamin-telegram-scheduler.service" 2>/dev/null || true

echo "[INFO] Reloading systemd..."
systemctl daemon-reload

echo "[INFO] Enabling and starting timer..."
systemctl enable --now benjamin-agentic-discussion.timer

echo "[INFO] Done. Verify with: systemctl status benjamin-agentic-discussion.timer --no-pager"



