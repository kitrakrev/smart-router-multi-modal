# AgentSafety — MedVisionRouter

**Task-Aware Medical Multimodal LLM Router**  
A10 Networks Hackathon — "Paying ATEN-tion to Agentic AI" (April 17-18, 2026)

## Quick Start

### On 192.168.50.221 (H200 — Medical Models)

```bash
cd /var/home/gpuuser/AgentSafety

# Install dependencies
pip3 install --user torch transformers accelerate sentence-transformers open-clip-torch fastapi uvicorn pyyaml httpx Pillow numpy

# Start medical model server (MedGemma-4B + LLaVA-Med-7B)
HF_TOKEN=$(cat ~/.cache/huggingface/token) python3 serve_models.py
# Runs on port 8001
```

### On 192.168.50.210 (H200 — Router + General Models)

```bash
cd /var/home/gpuuser/AgentSafety

# Install dependencies
pip3 install --user torch transformers accelerate sentence-transformers fastapi uvicorn pyyaml httpx Pillow numpy

# Terminal 1: Start general model server (Qwen2.5-7B + LLaVA-1.5-7B)
python3 serve_general.py
# Runs on port 8002

# Terminal 2: Start router + dashboard
python3 -m src.server --host 0.0.0.0 --port 8000
# Dashboard: http://192.168.50.210:8000/dashboard
# API Docs:  http://192.168.50.210:8000/docs
```

### If running on same machine (single GPU)

```bash
cd /var/home/gpuuser/AgentSafety
pip3 install --user torch transformers accelerate sentence-transformers fastapi uvicorn pyyaml httpx Pillow numpy

# Start router (works without model servers — routing-only mode)
python3 -m src.server --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000/dashboard
```

## What It Does

Routes medical queries to the optimal open-source model based on:
- **Specialty detection** (pathology, radiology, dermatology) via trained BioLORD embeddings
- **Medical image classification** via BioMedCLIP (histology, X-ray, dermoscopy)
- **Budget-aware model selection** (cheapest, quality-first, balanced, critical)
- **Per-model × per-specialty prompts** that auto-adapt from user feedback
- **Dynamic model registration** with capability probing

## Architecture

```
Query + Image
      │
      ├─ Text Embed (trained BioLORD) → specialty match (0.99 confidence)
      ├─ Vision Embed (BioMedCLIP) → image type classification
      │
      ▼
  Specialty Match → Model Select → Prompt Lookup → Forward to Model
      │
      ├─ pathology/radiology → LLaVA-Med-7B (H200)
      ├─ dermatology → MedGemma-4B (H200)
      ├─ code/reasoning → Qwen2.5-7B (A100/210)
      └─ general vision → LLaVA-1.5-7B (A100/210)
```

## Models (All Open-Source)

| Model | Params | Specializes In | GPU |
|-------|--------|---------------|-----|
| MedGemma-4B | 4B | Dermatology (vision+text) | 221 |
| LLaVA-Med-7B | 7B | Pathology, Radiology (vision+text) | 221 |
| Qwen2.5-7B | 7B | Code, Reasoning, General text | 210 |
| LLaVA-1.5-7B | 7B | General vision | 210 |

## Key Features

- **97.8% routing accuracy** on real medical images (PathVQA, VQA-RAD)
- **28ms routing latency** (6 parallel signals)
- **Trained embeddings** on 9K medical image dataset → 100% domain accuracy
- **Auto-adapt prompts** from user feedback (thumbs up/down → prompt refines)
- **Dynamic model add/remove/probe** at runtime — no restart
- **Degradation detection** — EMA stats auto-reroute away from slow models
- **Explainability** — "Why this path?" on every routing decision
- **4/4 models live** across 2 GPUs

## API Endpoints

```bash
# Route a query (routing + model inference)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Analyze this tissue biopsy"}]}'

# Routing only (no inference)
curl -X POST http://localhost:8000/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"query": "Chest X-ray findings"}'

# Explainability
curl -X POST http://localhost:8000/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "Skin lesion evaluation"}'

# Health
curl http://localhost:8000/health
```

## Dashboard (7 Tabs)

1. **Live Traces** — route queries, see model responses with latency breakdown
2. **Pipeline Flow** — interactive routing pipeline visualization
3. **Taxonomy** — specialty hierarchy with image type mapping
4. **Models** — edit costs, probe capabilities, simulate degradation
5. **Prompts** — per-model × per-specialty prompt matrix
6. **Sessions** — session history with specialty distribution
7. **Chat** — conversational interface with per-message routing

## Benchmark Results

| Metric | Score |
|--------|-------|
| Specialty accuracy (real images) | 97.8% |
| PathVQA (100 histology) | 100% |
| VQA-RAD (100 radiology) | 99% |
| Trained embeddings (4 domains) | 100% |
| Routing latency | 28ms |
| Unit tests | 67/67 pass |

## Run Tests

```bash
pip install pytest pytest-asyncio httpx
python3 -m pytest tests/ -v
python3 -m benchmarks.scenario_tests --verbose
python3 -m benchmarks.comprehensive_eval --max-samples 1000
```

## Team

**AgentSafety** — A10 Networks Hackathon 2026

## Dataset

Located at `data/` on both VMs (`/var/home/gpuuser/AgentSafety/data/`):

- `routing_dataset.csv` — 9,293 samples with image paths, specialty labels, dataset sources, train/test split
- `images/` — 9,293 medical images (pathology, radiology, dermatology, general)

| Domain | Count | Source | Image Types |
|--------|-------|--------|-------------|
| Pathology | 2,500 | PathVQA | Histology slides |
| Dermatology | 2,500 | Skin-Lesion-Dataset | Dermoscopy photos |
| General | 2,500 | VQAv2 | Everyday photos |
| Radiology | 1,793 | VQA-RAD | X-rays, CT, MRI |

Split: 7,434 train / 1,859 test

### Train Router Embeddings

```bash
cd /var/home/gpuuser/AgentSafety
python3 training/train_router_with_queries.py
# Pre: 87.5% → Post: 100% accuracy
# Saves to trained_router_v2/
```

## GitHub

```
https://github.com/kitrakrev/smart-router-multi-modal
Branch: v2-medical
```
