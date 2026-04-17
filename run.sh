#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=================================="
echo "  LLM Router — Setup & Launch"
echo "=================================="

# Launch server with uv (auto-installs deps)
echo "[1/1] Starting server on http://localhost:8000"
echo ""
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Eval API:   POST http://localhost:8000/v1/eval"
echo ""
uv run uvicorn server:app --host 0.0.0.0 --port 8000
