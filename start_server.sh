#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LungCancerDX – One-Shot Server Launcher
# Usage: bash start_server.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🫁  LungCancerDX  —  Starting Server      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 not found. Please install Python 3.9+."
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, cv2, PIL" 2>/dev/null || {
    echo "⚠️  Missing packages detected. Installing..."
    pip install -r requirements.txt
}

echo ""
echo "🚀 Starting FastAPI server on http://localhost:8000"
echo "   → Swagger docs: http://localhost:8000/docs"
echo "   → Frontend:     http://localhost:8000"
echo ""
echo "   Press Ctrl+C to stop."
echo ""

python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
