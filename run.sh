#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=================================="
echo "  LLM Router — Quick Start"
echo "=================================="
echo ""
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Eval API:   POST http://localhost:8000/v1/eval"
echo ""
uv run python -m src.server --host 0.0.0.0 --port 8000
