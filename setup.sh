#!/bin/bash
# MedVisionRouter v2 setup and run script
# Installs dependencies, downloads models (BioMedCLIP + MiniLM), starts server
set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  MedVisionRouter v2 — Setup"
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
uv sync

# Step 3: Pre-download embedding models
echo "[3/4] Checking embedding models..."
uv run python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('  MiniLM-L6-v2 loaded (384-dim embeddings)')
" 2>/dev/null || echo "  MiniLM download will happen on first request"

uv run python3 -c "
try:
    import open_clip
    open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    print('  BioMedCLIP loaded')
except Exception as e:
    print(f'  BioMedCLIP skipped: {e}')
" 2>/dev/null || echo "  BioMedCLIP download will happen on first request"

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
uv run python -m src
