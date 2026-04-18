#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  Smart Multi-Modal LLM Router — Setup"
echo "============================================"
echo ""

# Step 1: Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[1/4] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/4] uv already installed"
fi

# Step 2: Install Python dependencies
echo "[2/4] Installing dependencies..."
uv sync 2>/dev/null || uv pip install -r requirements.txt
echo "  Dependencies installed"

# Step 3: Verify sentence-transformers (required for embeddings)
echo "[3/4] Checking embedding model..."
uv run python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('  MiniLM-L6-v2 loaded (384-dim embeddings)')
" 2>/dev/null || {
    echo "  Installing sentence-transformers..."
    uv pip install sentence-transformers
    uv run python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('  MiniLM-L6-v2 loaded')"
}

# Step 4: Start server
echo "[4/4] Starting server..."
echo ""
echo "============================================"
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Eval API:   POST http://localhost:8000/v1/eval"
echo "  WebSocket:  ws://localhost:8000/ws/traces"
echo "============================================"
echo ""
uv run python -m src.server --host 0.0.0.0 --port 8000
