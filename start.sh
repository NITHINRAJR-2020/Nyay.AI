#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load env if present
if [ -f .env ]; then
  echo "[start] Loading .env"
  export $(grep -v '^#' .env | xargs)
fi

# Activate virtualenv if present (FIXED)
if [ -d "venv" ]; then
  echo "[start] Activating virtual environment"
  source venv/Scripts/activate
fi

echo ""
echo "  ⚖️  Nyay.AI — Legal Case Intelligence System"
echo "  ─────────────────────────────────────────────"
echo "  URL: http://localhost:7000"
echo ""

cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 7000
