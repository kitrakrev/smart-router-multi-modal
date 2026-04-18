# MedVisionRouter v2

**Task-aware medical multimodal LLM router** — routes clinical queries to the optimal open-source model based on specialty detection, medical image classification, budget strategy, and real-time model reputation.

**A10 Networks Hackathon** — "Paying ATEN-tion to Agentic AI" (April 17-18, 2026)

---

## Features

### Core Routing
- **Two-level specialty routing**: general vs medical → specific specialty (pathology, radiology, dermatology, etc.)
- **6 parallel signals** extracted per query in ~28ms (text, vision, complexity, safety, tools, modality)
- **Dual-embedding ensemble**: BioLORD-2023 (medical) + BGE-small-en-v1.5 (general) — max similarity across both
- **Vision routing**: BioMedCLIP classifies medical image types (histology, X-ray, CT, MRI, dermoscopy, fundus, ECG)
- **Non-medical image detection**: low vision confidence → routes to general vision model (LLaVA-1.6)
- **Hierarchical specialty taxonomy** with alias deduplication ("Pathologies" → "pathology")

### Budget & Model Selection
- **5 budget strategies**: `cheapest_capable`, `quality_first`, `balanced`, `critical`, `performance_weighted`
- **Critical auto-detect**: emergency keywords / high complexity → auto-upgrade to best model + 4096 reasoning tokens
- **Dynamic model registration**: add/remove/probe models at runtime via API
- **Model cost editing**: update costs live from dashboard, routing adapts instantly

### Adaptive Learning & Reputation
- **Per-user model reputation**: one user's bad experience only affects that user's routing
- **Global model reputation**: multiple users give bad feedback → model score drops for everyone
- **EMA stats tracker**: latency + error rate smoothing, auto-disable degraded models
- **Simulated degradation**: inject latency spikes (60s window) for demo, model stays enabled but router prefers alternatives
- **Auto-recovery**: disabled models re-checked after 2-min cooldown, re-enabled if healthy
- **User feedback endpoint**: `POST /v1/feedback` with rating 0.0-1.0

### Ambiguity Resolution
- **Embedding confidence check**: similarity < 0.35 → query is ambiguous
- **LLM fallback classifier**: Qwen2.5-0.5B-Instruct (500M params, ~200ms) classifies ambiguous queries into domain
- **Trained router embeddings**: BioLORD fine-tuned on 800-image medical dataset → 100% accuracy on 4 domains

### Prompts & DSPy
- **Per-model × per-specialty prompts**: medgemma gets different pathology prompt than medalpaca
- **3 template types**: specialist (concise), generalist (systematic), reasoning (chain-of-thought)
- **12 model-specific overrides** in config (medgemma×4, medalpaca×4, llama×2, qwen×2)
- **DSPy MIPROv2 optimizer**: auto-optimize prompts per model×specialty pair
- **Priority chain**: DSPy override > manual override > auto-generated template

### Safety & Tools
- **Two-layer safety gate**: regex fast-pass (<1ms) + contrastive embedding (~3ms) for jailbreak + PHI
- **100% safety accuracy**: 0 false positives, 0 false negatives on 500-sample benchmark
- **5 medical tools**: drug_interaction_check, clinical_guideline, lab_reference, dosage_calculator, icd_code_lookup
- **Embedding-based tool detection**: MiniLM cosine sim to tool exemplars

### Explainability
- **Inline "Why this path?"** on every trace in dashboard — signal-by-signal breakdown
- **Signal flow visualization**: text match → specialty → capability filter → budget strategy → model
- **Cost comparison**: estimated cost vs GPT-4o baseline with savings percentage
- **`/v1/explain` API endpoint**: full explainability JSON + human-readable text

### Vision & Multimodal
- **BioMedCLIP** (196M params, PubMedBERT + ViT-B/16) for medical image classification
- **CLIP ViT-B-32 fallback** when open_clip unavailable
- **8 medical image types**: histology, xray, ct, mri, dermoscopy, fundus, clinical_photo, ecg
- **Non-medical image routing**: low confidence (< 0.5) → general vision model
- **End-to-end image pipeline**: base64 image → BioMedCLIP classify → specialty → model → inference

### Dashboard (7 tabs)
- **Live Traces**: real-time routing decisions via WebSocket, inline explainability, model response display
- **Pipeline Flow**: 11-node interactive SVG with click-to-expand technical details
- **Taxonomy**: hierarchical specialty tree with image type mapping
- **Models**: model cards with cost editing, degradation simulation, capability probing, radar chart
- **Prompts**: model_type × specialty matrix + model-specific override table
- **Sessions**: session list with specialty distribution and trace timeline
- **Chat**: session-based conversations with per-message routing metadata and "Why?" button

### Multi-GPU Architecture
- **A100 (80GB)**: Router + embedding signals + general models (Llama-3.1-8B, Qwen2.5-7B, LLaVA-1.6-7B)
- **H200 (35GB MIG)**: Medical specialist models (MedGemma-4B, MedAlpaca-7B)
- **A10-gpu (18GB MIG)**: Training (router embeddings, fallback classifier)
- **SSH tunnel bridging**: A100↔H200 via local relay when on different networks

### Testing & Benchmarks
- **67 unit tests**: 100% pass, 74% code coverage
- **5K comprehensive benchmark**: 97.2% specialty accuracy, 100% safety, 90% critical detection
- **400 real image benchmark**: 97.8% overall (PathVQA 100%, VQA-RAD 99%, PubMedQA 96%)
- **22 scenario tests**: medical text, vision, tools, safety, general, ambiguous, complex, emergency
- **Frontend→backend verification**: 0 API mismatches across 16 fetch() calls

---

## Quick Start

```bash
git clone https://github.com/kitrakrev/smart-router-multi-modal.git
cd smart-router-multi-modal
git checkout v2-medical
chmod +x setup.sh && ./setup.sh

# Dashboard: http://localhost:8000/dashboard
# API Docs:  http://localhost:8000/docs
```

---

## Embedding Models Used

| Model | Purpose | Params | Latency |
|-------|---------|--------|---------|
| BioLORD-2023 | Medical text specialty matching | 33M | ~12ms |
| BGE-small-en-v1.5 | General text specialty matching | 33M | ~12ms |
| BioMedCLIP | Medical image type classification | 196M | ~20ms |
| MiniLM-L6-v2 | Complexity, safety, tools signals | 22M | ~8ms |
| Qwen2.5-0.5B | Ambiguity fallback classifier | 500M | ~200ms |

## LLM Models Served

| Model | GPU | Type | Capabilities |
|-------|-----|------|-------------|
| MedGemma-4B | H200 | specialist | vision, medical, text |
| MedAlpaca-7B | H200 | specialist | medical, text |
| Llama-3.1-8B | A100 | generalist | text, code, reasoning |
| Qwen2.5-7B | A100 | reasoning | text, reasoning, code, tools |
| LLaVA-1.6-7B | A100 | generalist | vision, text |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible routing + model inference |
| `POST` | `/v1/eval` | Routing decision only (no inference) |
| `POST` | `/v1/explain` | Routing + full explainability breakdown |
| `POST` | `/v1/feedback` | User feedback on model response quality |
| `GET` | `/v1/models` | List models with runtime stats |
| `POST` | `/v1/models` | Register new model (with api_base) |
| `PUT` | `/v1/models/{name}` | Update model fields (cost, quality, etc.) |
| `DELETE` | `/v1/models/{name}` | Remove model |
| `POST` | `/v1/models/{name}/probe` | Probe model capabilities (real inference) |
| `POST` | `/v1/models/{name}/enable` | Enable model |
| `POST` | `/v1/models/{name}/disable` | Disable model |
| `POST` | `/v1/models/discover` | Auto-discover from OpenAI-compatible endpoint |
| `POST` | `/v1/stats/simulate` | Inject latency degradation (duration_s) |
| `GET` | `/v1/stats/models` | Runtime EMA stats for all models |
| `GET` | `/v1/specialties` | Taxonomy tree + specialty list |
| `GET` | `/v1/prompts/matrix` | Full prompt matrix (type × specialty + overrides) |
| `GET` | `/v1/sessions` | List recent sessions |
| `GET` | `/v1/sessions/{id}/traces` | Session trace timeline |
| `GET` | `/v1/users/{id}/stats` | Per-user routing stats |
| `GET` | `/v1/tools` | List medical tools |
| `POST` | `/v1/tools/execute` | Execute a tool directly |
| `GET` | `/v1/config` | Current config (all YAMLs) |
| `POST` | `/v1/config/reload` | Hot-reload config from disk |
| `GET` | `/v1/traces` | Recent traces |
| `WS` | `/ws/traces` | WebSocket live trace streaming |
| `GET` | `/dashboard` | Interactive dashboard |
| `GET` | `/health` | Health check |

---

## Routing Logic

```
Query + optional image
         │
         ▼
   ┌──────────────────────────────────────┐
   │  6 SIGNALS IN PARALLEL (~28ms)       │
   │  Text (BioLORD+BGE) │ Vision (CLIP)  │
   │  Complexity          │ Safety        │
   │  Tools               │ Modality      │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  CRITICAL AUTO-DETECT                │
   │  emergency keywords / complexity>0.8 │
   │  → auto-upgrade to best model        │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  SAFETY GATE                         │
   │  risk > 0.8 → BLOCK                  │
   │  risk 0.3-0.8 → FLAG                 │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  SPECIALTY MATCH                     │
   │  text embedding + vision image type  │
   │  → taxonomy lookup + alias dedup     │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  AMBIGUITY CHECK                     │
   │  similarity < 0.35?                  │
   │  → call Qwen2.5-0.5B LLM classifier │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  MODEL SELECTION                     │
   │  base_quality + global_rep + user_rep│
   │  critical → always best              │
   │  balanced → weighted scoring         │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  PROMPT LOOKUP                       │
   │  DSPy > manual > auto template       │
   │  per-model × per-specialty           │
   └──────────┬───────────────────────────┘
              │
   ┌──────────▼───────────────────────────┐
   │  FORWARD TO MODEL_RUNNER             │
   │  A100 (general) or H200 (medical)    │
   │  Record stats, broadcast via WS      │
   └──────────────────────────────────────┘
```

---

## Benchmark Results

| Metric | Score |
|--------|-------|
| Specialty routing (5K synthetic) | 97.2% |
| Specialty routing (400 real images) | 97.8% |
| PathVQA (100 histology images) | 100% |
| VQA-RAD (100 radiology images) | 99% |
| PubMedQA (200 clinical text) | 96% |
| Safety gate (500 samples) | 100% |
| Critical auto-detection | 90% |
| Tool detection | 75% |
| Routing latency (avg) | 28ms |
| Routing latency (P95) | 45ms |
| Trained embeddings (4 domains) | 100% |
| Ambiguous → general_medicine | 100% |
| Unit tests | 67/67 pass |
| Code coverage | 74% |

---

## Project Structure

```
router-prototype/
├── src/
│   ├── server.py              # FastAPI server, 31 endpoints
│   ├── router.py              # Two-level routing engine
│   ├── explainability.py      # Signal-by-signal routing explanations
│   ├── fallback_classifier.py # Qwen2.5-0.5B ambiguity classifier
│   ├── signals/               # 6 parallel signal extractors
│   │   ├── text.py            #   BioLORD + BGE dual ensemble
│   │   ├── vision.py          #   BioMedCLIP image classification
│   │   ├── complexity.py      #   Contrastive hard/easy
│   │   ├── safety.py          #   Regex + embedding jailbreak/PHI
│   │   ├── tools.py           #   Medical tool detection
│   │   └── modality.py        #   text_only / vision / multimodal
│   ├── taxonomy/              # Specialty hierarchy + alias dedup
│   ├── registry/              # Model CRUD + EMA stats + probing
│   ├── prompts/               # Per-model × per-specialty templates
│   ├── memory/                # User memory + session tracking
│   └── tools/                 # Medical tool executor
├── config/                    # 6 YAML config files
├── dashboard/
│   └── index.html             # 7-tab interactive dashboard
├── serve_models.py            # H200 medical model server (port 8001)
├── serve_general.py           # A100 general model server (port 8002)
├── training/                  # Router embedding + fallback training
├── benchmarks/                # 5K eval, real images, scenarios
├── tests/                     # 67 unit tests
├── optimize/                  # DSPy optimizer + evaluator
└── pyproject.toml
```

---

## License

Apache-2.0
