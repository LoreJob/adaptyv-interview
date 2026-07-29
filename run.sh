#!/usr/bin/env bash
# One-command launcher: sets up both halves and starts them.
# Backend on http://localhost:8000, frontend on http://localhost:5173.
# Stop with Ctrl-C (kills both).
set -euo pipefail
cd "$(dirname "$0")"

# --- Python backend ---
if [ ! -d .venv ]; then
  echo "==> creating virtualenv"
  python3 -m venv .venv
fi
echo "==> installing Python deps"
.venv/bin/pip install -q -r requirements.txt

# --- React frontend ---
echo "==> installing frontend deps"
( cd frontend && npm install --silent )

# --- launch both ---
echo "==> starting backend on :8000"
.venv/bin/uvicorn src.api:app --port 8000 &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT

echo "==> starting frontend on :5173  (open http://localhost:5173)"
cd frontend && npm run dev
