# MedVisionRouter v2

**Medical multimodal router with vision, taxonomy-based specialty routing, DSPy-optimized prompts, and budget-aware model selection.** Routes clinical queries to the optimal medical model based on specialty detection, image type classification, and contrastive embeddings.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-kitrakrev%2Fsmart--router--embeddings-yellow.svg)](https://huggingface.co/kitrakrev/smart-router-embeddings)

---

## What This Is

A production-grade medical LLM routing layer that sits between clinical applications and multiple medical/general AI models. The router:

1. **Extracts 9 signals** from the incoming query (safety, PII, vision, domain, complexity, etc.)
2. **Two-level routing**: general vs medical, then specialty classification (pathology, radiology, cardiology, etc.)
3. **Matches** queries against decision exemplar centroids using sentence embeddings (vLLM-SR style)
4. **Selects** the cheapest capable model that satisfies the matched specialty's requirements
5. **Applies** DSPy-optimized prompts, safety guardrails, budget constraints, and tool augmentation

The result: a pathology VQA question with an image gets routed to MedGemma-4B with a pathologist system prompt; a simple drug interaction query goes to MedAlpaca with a pharmacology template and the `drug_interaction_check` tool -- all automatically, in under 15ms.

---

## Architecture

```
                           MedVisionRouter v2 Pipeline
                           ===========================

Query + Image ──> Safety Gate ──> Signal Extractors (9 parallel) ──> Two-Level Router ──> Model
                      |                  |                                  |               |
                  regex + embed     keyword, domain,               Level 1: general        budget-aware
                  jailbreak/PHI     complexity, language,            vs medical             selection
                  block / warn      safety, PII, vision,                |                     |
                                    tool, modality              Level 2: specialty         cheapest_capable
                                         |                      pathology                  quality_first
                                    MiniLM-L6-v2               radiology                   balanced
                                    embeddings                  cardiology                  performance_weighted
                                                                dermatology
                                                                ophthalmology
                                                                emergency
                                                                pharmacology
                                                                general_medicine
                                                                      |
                                                               DSPy-optimized prompt
                                                               + tool attachment
                                                               + inference config

Config files:
  config/taxonomy.yaml        -- specialty tree (medical + general)
  config/models.yaml          -- model pool (local + API)
  config/prompt_templates.yaml -- per-specialty prompts + DSPy overrides
  config/safety.yaml          -- regex + contrastive safety patterns
  config/tools.yaml           -- tool definitions (drug_interaction, ICD codes, etc.)
  config/probes.yaml          -- model capability probes
  config/config.yaml          -- main routing config (decisions, budget, safety rules)
```

---

## Quick Start

```bash
# Clone and enter the project
git clone <repo-url> router-prototype
cd router-prototype

# One-command install and launch
chmod +x setup.sh && ./setup.sh

# Open the dashboard
open http://localhost:8000/dashboard
```

Or manually:

```bash
uv sync
uv run python -m src.server --host 0.0.0.0 --port 8000
```

Test the eval endpoint (no API keys needed):

```bash
# Medical text query
curl -s -X POST http://localhost:8000/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are the symptoms of diabetes?"}]}' \
  | python -m json.tool

# Medical query that should route to pathology
curl -s -X POST http://localhost:8000/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Analyze this biopsy slide for adenocarcinoma markers"}]}' \
  | python -m json.tool
```

---

## How Routing Works

### Level 1: General vs Medical

The domain signal (MiniLM embedding similarity against domain exemplars) classifies incoming queries as `medical`, `code`, `science`, `creative`, `math`, `legal`, or `general`. Medical queries proceed to specialty routing; general queries use the standard decision matcher.

### Level 2: Specialty Classification

For medical queries, the router matches against a taxonomy tree defined in `config/taxonomy.yaml`:

```
medical/
  cardiology/       (ecg, echocardiogram)
  radiology/        (xray, ct, mri)
  pathology/        (histology, cytology, gross_specimen)
  dermatology/      (dermoscopy, clinical_photo)
  ophthalmology/    (fundus, oct)
  emergency/
  pharmacology/
  general_medicine/

general/
  code/
  reasoning/
  creative/
  simple_qa/
```

Each specialty has associated image types, enabling vision-based routing (e.g., an uploaded histology image automatically routes to pathology).

### Model Selection

Each routing decision specifies:
- `require`: capabilities the model must have (e.g., `[vision, medical]`)
- `strategy`: budget strategy (`cheapest_capable`, `quality_first`, `balanced`, `performance_weighted`)

The router filters models from `config/models.yaml` and applies the strategy.

### Prompt Selection

`config/prompt_templates.yaml` defines per-specialty prompts (e.g., "You are a board-certified pathologist. Analyze tissue morphology..."). DSPy-optimized overrides are stored in the `overrides` section and take precedence when available.

---

## Config Reference

### taxonomy.yaml

Defines the two-level specialty tree with image type associations.

### models.yaml

Model pool with capabilities, costs, and quality scores:

| Model | Type | Capabilities | Cost/1K out |
|-------|------|-------------|-------------|
| medgemma-4b | specialist | vision, medical, text | $0.00 (local) |
| llava-med-7b | specialist | vision, medical, text | $0.00 (local) |
| medalpaca-7b | specialist | medical, text | $0.00 (local) |
| llama-3.1-8b | generalist | text, code, reasoning | $0.00 (local) |
| gpt-4o | reasoning | vision, text, reasoning, tools, medical | $10.00 |

### prompt_templates.yaml

Three model types (`specialist`, `generalist`, `reasoning`) with per-specialty customizations. The `overrides` section stores DSPy-optimized prompts with few-shot demos and accuracy metadata.

### safety.yaml

Regex patterns (jailbreak, PHI) and contrastive embedding exemplars (unsafe vs safe). Safety signal uses both layers: fast regex first, then embedding similarity for paraphrased attacks.

### tools.yaml

Tool definitions with exemplar triggers: `drug_interaction_check`, `clinical_guideline`, `lab_reference`, `dosage_calculator`, `icd_code_lookup`.

### probes.yaml

Specialty-specific capability probes with expected keywords for model validation.

---

## Benchmarks

### PathVQA (Pathology Vision QA)

5K samples from 32K total. Tests vision detection, pathology specialty routing, and vision-capable model selection.

```bash
python -m benchmarks.pathvqa --max-samples 5000 --verbose
```

### PubMedQA (Clinical Text QA)

1K labeled samples. Tests medical domain detection, medical model selection, and reasoning routing.

```bash
python -m benchmarks.pubmedqa --max-samples 1000 --verbose
```

### Scenario Tests

24 curated scenarios covering medical text, medical vision, drug interactions, safety, general queries, ambiguous inputs, complex reasoning, and emergencies. Runs entirely offline (no model inference).

```bash
python -m benchmarks.scenario_tests --verbose
```

### RouterArena

8,400 queries with ground truth from 3 models.

| Metric | Value |
|--------|-------|
| Routing accuracy (809 overlap queries) | **86.8%** |
| Average routing latency | 15ms |
| Cost per 1K queries | $0.25 |

### VL-RouterBench

30K+ vision-language samples across 14 datasets and 17 VLMs.

| Metric | Value |
|--------|-------|
| Routing accuracy | **83.8%** |
| Rank score | 0.80 |

### Running All Benchmarks

```bash
# Routing-only evaluation (no model server needed)
python -m optimize.evaluate --all --mode routing-only

# End-to-end evaluation (requires model server)
python -m optimize.evaluate --all --mode end-to-end --api-base http://localhost:8000/v1
```

---

## DSPy Optimization

The DSPy optimizer fine-tunes prompts per (model, specialty) pair using MIPROv2. It loads a medical dataset, runs the optimizer to find optimal instructions and few-shot demos, then saves results to `config/prompt_templates.yaml`.

### Single Pair

```bash
python -m optimize.dspy_optimizer \
  --model medgemma-4b \
  --specialty pathology \
  --dataset pathvqa \
  --api-base http://localhost:8000/v1
```

### All Pairs

```bash
python -m optimize.dspy_optimizer --all
```

### How It Works

1. Load 100 train + 50 eval examples from the target dataset
2. Configure DSPy with the target model (via OpenAI-compatible API)
3. Define a `MedicalQA` signature: `question + context -> answer`
4. Run MIPROv2 with 10 candidates and up to 3 bootstrapped demos
5. Evaluate baseline vs optimized accuracy (token F1 + exact match)
6. Save optimized prompt + demos to YAML with metadata

Results are saved to `optimize/results/` and merged into `config/prompt_templates.yaml` overrides.

---

## Training Embeddings

### Medical Routing Embeddings

Fine-tune MiniLM-L6-v2 on medical datasets for better routing accuracy:

```bash
# Full training (works on CUDA, MPS, or CPU)
python -m benchmarks.train_embeddings

# Custom settings
python -m benchmarks.train_embeddings \
  --batch-size 256 \
  --epochs 3 \
  --output-dir models/med-routing-embeddings

# Skip BioMedCLIP classifier
python -m benchmarks.train_embeddings --skip-clip
```

Data sources (5K each, loaded partially):
- **PathVQA**: pathology visual questions
- **PubMedQA**: clinical text questions
- **LMSYS 55K**: general queries (for contrast)

Method: MultipleNegativesRankingLoss with in-batch negatives. Same specialty = positive pair, cross-specialty = negative. Includes medical domain exemplars for specialty-specific clustering.

Optional: BioMedCLIP-based image type classifier (histology vs xray vs dermoscopy vs fundus vs clinical_photo).

### General Routing Embeddings

The existing training script in `training/train_routing_embeddings.py` trains on LMSYS 55K for general task-type clustering (code, math, creative, QA, etc.).

---

## API Reference

### Chat and Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat with automatic routing |
| `POST` | `/v1/eval` | Classify query, return signals + routing decision (no API key) |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/models` | List all registered models |
| `POST` | `/v1/models` | Register a new model |
| `PUT` | `/v1/models/{name}` | Update a model |
| `DELETE` | `/v1/models/{name}` | Remove a model |
| `GET` | `/v1/models/{name}/health` | Health check a model endpoint |
| `POST` | `/v1/models/discover` | Auto-discover from OpenAI-compatible endpoint |

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/config` | Return current in-memory config |
| `PUT` | `/v1/config` | Replace full config (hot reload) |
| `PATCH` | `/v1/config/rules` | Update routing rules only |
| `PATCH` | `/v1/config/budget` | Update budget constraints only |
| `POST` | `/v1/config/reload` | Reload config.yaml from disk |

### Traces and Stats

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/traces` | Recent traces (default 50) |
| `GET` | `/v1/trace/{id}` | Single trace by ID |
| `GET` | `/v1/stats` | Aggregate routing statistics |
| `GET` | `/v1/stats/models` | Runtime model performance stats |
| `WS` | `/ws/traces` | WebSocket live trace streaming |

### Tools and Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/tools` | List available tools |
| `POST` | `/v1/tools/execute` | Execute a tool directly |
| `GET` | `/dashboard` | Unified HTML dashboard |
| `GET` | `/docs` | Interactive Swagger API docs |

---

## Dashboard

The dashboard at `/dashboard` provides 7 tabs:

1. **Live Traces** -- real-time feed of every routing decision via WebSocket, with signal breakdown radar charts
2. **Pipeline Flow** -- visual representation of the routing pipeline for each query
3. **Models** -- registered models with capabilities, costs, quality scores, and health status
4. **Stats** -- aggregate metrics: total requests, model distribution, latency percentiles, cost per 1K
5. **Sessions** -- per-session query history and adaptive routing updates
6. **Config** -- live view and editor for YAML config with hot-reload
7. **Datasets** -- benchmark dataset explorer for RouterArena and VL-RouterBench results

---

## Project Structure

```
router-prototype/
├── src/                              # Core application
│   ├── __init__.py
│   ├── __main__.py                   # Entry point: python -m src
│   ├── server.py                     # FastAPI server, all endpoints
│   ├── router.py                     # Decision engine + model selection
│   ├── signals.py                    # 9 signal extractors (parallel)
│   ├── signals/vision.py             # Vision-specific signal logic
│   ├── models.py                     # Dynamic model registry
│   ├── models_api.py                 # Model CRUD REST API
│   ├── tracer.py                     # Request tracing + session tracking
│   ├── tools.py                      # Tool executor
│   ├── task_classifier.py            # LMSYS multi-head classifier
│   └── registry/                     # Model registry internals
│
├── config/                           # Configuration
│   ├── config.yaml                   # Main config (decisions, budget, safety)
│   ├── taxonomy.yaml                 # Medical specialty tree
│   ├── models.yaml                   # Model pool (local + API)
│   ├── prompt_templates.yaml         # Per-specialty prompts + DSPy overrides
│   ├── safety.yaml                   # Regex + contrastive safety patterns
│   ├── tools.yaml                    # Tool definitions + exemplar triggers
│   ├── probes.yaml                   # Model capability probes
│   └── generated/                    # Auto-generated benchmark configs
│
├── optimize/                         # DSPy optimization pipeline
│   ├── __init__.py
│   ├── dspy_optimizer.py             # MIPROv2 prompt optimization per model x specialty
│   ├── evaluate.py                   # Benchmark runner (routing-only + end-to-end)
│   └── results/                      # Optimization results (YAML)
│
├── benchmarks/                       # Benchmark scripts + results
│   ├── pathvqa.py                    # PathVQA routing benchmark (5K samples)
│   ├── pubmedqa.py                   # PubMedQA routing benchmark (1K samples)
│   ├── scenario_tests.py             # 24 curated scenario tests (offline)
│   ├── train_embeddings.py           # Medical routing embedding trainer
│   ├── benchmark_routerarena.py      # RouterArena evaluation
│   ├── benchmark_vl_routerbench.py   # VL-RouterBench evaluation
│   ├── generate_benchmark_config.py  # Auto-config generator
│   └── results/                      # Benchmark output JSON
│
├── training/                         # General training scripts
│   ├── finetune_lmsys.py             # LMSYS 55K multi-head classifier
│   ├── finetune_router.py            # RouterArena classifier
│   └── train_routing_embeddings.py   # General contrastive embeddings
│
├── dashboard/                        # Frontend UI
│   ├── index.html                    # Unified 7-tab dashboard
│   ├── dashboard.html
│   └── trace_dashboard.html
│
├── models/                           # Trained model weights
│   ├── med-routing-embeddings/       # Medical fine-tuned embeddings
│   ├── lmsys-task-classifier/
│   └── routerarena-classifier/
│
├── presentation/                     # Hackathon presentation
│   ├── slides.pptx
│   └── DEMO_SCRIPT.md
│
├── setup.sh                          # One-command install + launch
├── run.sh                            # Quick start
├── pyproject.toml                    # Project metadata + dependencies
└── uv.lock                          # Lockfile
```

---

## License

Apache-2.0
