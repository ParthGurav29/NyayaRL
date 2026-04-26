#!/bin/bash
# scripts/run_demo.sh — NyayaRL demo launcher

set -e

cd "$(dirname "$0")/.."

PYTHON="python3"
if [[ -x "venv/bin/python" ]]; then
  PYTHON="venv/bin/python"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║           NyayaRL — Legal RL Agent Demo          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. Verify symlink
RESOLVED=$($PYTHON -c "import os; print(os.path.realpath('checkpoints/latest.pt'))")
echo "▸ Checkpoint : $RESOLVED"

if [[ "$RESOLVED" != *"step_000299"* ]]; then
  echo "❌ Wrong checkpoint loaded. Expected step_000299.pt"
  echo "   Run: rm checkpoints/latest.pt && ln -s step_000299.pt checkpoints/latest.pt"
  exit 1
fi

# 2. Run heldout eval and print the number
echo "▸ Running heldout eval..."
$PYTHON training/eval_heldout.py --checkpoint checkpoints/latest.pt
echo ""

# 3. Launch Gradio
echo "▸ Launching demo UI..."
echo ""
# Kill any stale Gradio on these ports (macOS lsof doesn't accept ":p1,:p2")
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
lsof -ti :7861 | xargs kill -9 2>/dev/null || true

# Pick a free port (prefer 7861, then 7862..7870) unless user set GRADIO_SERVER_PORT.
if [[ -z "${GRADIO_SERVER_PORT:-}" ]]; then
  GRADIO_SERVER_PORT=$($PYTHON - <<'PY'
import socket
for port in range(7861, 7871):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            print(port)
            raise SystemExit(0)
print(7861)
PY
)
fi
export GRADIO_SERVER_PORT

$PYTHON gradio_app.py

