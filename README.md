# Smart Router Multi-Modal

**Multi-modal LLM router with vision, tools, and real-time tracing.** Routes queries to the optimal model based on embedding similarity to decision exemplars -- budget-aware, adaptive, observable.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-kitrakrev%2Fsmart--router--embeddings-yellow.svg)](https://huggingface.co/kitrakrev/smart-router-embeddings)

---

## What This Is

A production-ready LLM routing layer that sits between your application and multiple LLM providers. Instead of hardcoding which model handles which query, the router:

1. Extracts 9 signals from the incoming query (complexity, domain, safety, vision, tools, etc.)
2. Matches the query against decision exemplar centroids using sentence embeddings (vLLM-SR style)
3. Selects the cheapest capable model that satisfies the matched decision's requirements
4. Applies safety guardrails, budget constraints, and adaptive health-aware routing

The result: complex math goes to reasoning models, simple QA goes to tiny local models, vision queries go to multimodal models -- all automatically, in under 15ms.

---

## Key Features

- **9 signal extractors** -- keyword, domain, complexity, language, safety, PII, vision, tool, modality
- **Embedding-based decision matching** -- vLLM-SR style cosine similarity to decision exemplar centroids
- **Budget-aware model selection** -- strategies: `cheapest_capable`, `quality_first`, `balanced`, `performance_weighted`
- **Adaptive routing** -- EMA-based health tracking, auto-disable degraded models, auto-recover
- **Dynamic model registry** -- add/remove/discover models via REST API at runtime
- **Real-time dashboard** -- 7 tabs: traces, pipeline, models, stats, sessions, config, datasets
- **Image upload + vision routing** -- detects image content and routes to vision-capable models
- **Tool execution** -- 11 built-in tools (port scan, CVE lookup, firewall, vulnerability scan, etc.)
- **Session/user tracking** -- per-session and per-user routing history and stats
- **Hot-reloadable YAML config** -- update routing rules without restarting the server
- **OpenAI-compatible API** -- drop-in replacement for `/v1/chat/completions`

---

## Architecture

```
Query ──> Guardrails ──> Signal Extractors ──> Decision Matcher ──> Capability Filter ──> Budget Selection ──> Model
              |               |                      |                    |                     |
         safety rules    9 parallel:            embedding cosine     require: [vision]     cheapest_capable
         jailbreak       - keyword              similarity to        require: [tools]      quality_first
         PII             - domain (MiniLM)      decision exemplar    require: [reasoning]  balanced
                         - complexity           centroids            require: [text]       performance_weighted
                         - language                  |
                         - safety              matched decision
                         - PII                 config (temp,
                         - vision              max_tokens, etc.)
                         - tool need
                         - modality
```

---

## Quick Start

```bash
# Clone the repo
git clone <repo-url> smart-router-multi-modal
cd smart-router-multi-modal

# One-command install + launch
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
curl -s -X POST http://localhost:8000/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Prove sqrt(2) is irrational"}]}' | python -m json.tool
```

---

## How Routing Works

The router uses a 3-layer approach inspired by vLLM-SR:

### Layer 1: WHAT task? (Decision Matching)

Each "decision" in `config/config.yaml` defines a task type (e.g., `complex_reasoning`, `code_generation`, `vision_task`) with a list of exemplar queries. At startup, the router encodes all exemplars using `all-MiniLM-L6-v2` (384-dim sentence embeddings) and computes a centroid for each decision.

At inference time, the incoming query is embedded and matched against all centroids via cosine similarity. The highest-scoring decision wins. No threshold cutoff -- the embedding match handles ALL queries.

### Layer 2: WHICH model? (Capability + Budget)

Each decision specifies:
- `require`: list of capabilities the model must have (e.g., `[vision]`, `[reasoning, code]`)
- `strategy`: how to pick among capable models (`cheapest_capable`, `quality_first`, `balanced`, `performance_weighted`)

The router filters the model pool to those with ALL required capabilities, then applies the budget strategy.

### Layer 3: HOW to call? (Inference Config)

Each decision includes an inference config: `temperature`, `max_tokens`, `enable_reasoning`, `thinking_tokens`. This config is passed through to the model call, ensuring reasoning tasks get low temperature and high token limits while creative tasks get high temperature.

---

## Configuration

The main config is at `config/config.yaml`. Structure:

```yaml
models:
  - name: gpt-4o
    provider: openai
    cost_per_1k_input: 2.50        # USD per 1K input tokens
    cost_per_1k_output: 10.00      # USD per 1K output tokens
    avg_latency_ms: 800            # baseline latency
    capabilities: [vision, tools, reasoning, text]
    quality_score: 0.92            # 0-1, used by quality_first strategy

  - name: qwen-3b-local
    provider: local
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.003
    avg_latency_ms: 50
    capabilities: [text, general, fast]
    quality_score: 0.45

routing:
  budget:
    max_cost_per_query: 0.01       # USD hard cap
    strategy: cheapest_capable     # default global strategy
    quality_threshold: 0.7         # minimum quality for quality_first

  decisions:
    - name: complex_reasoning
      description: "Complex math, proofs, multi-step analysis"
      exemplars:                   # queries that define this decision
        - "prove that sqrt(2) is irrational"
        - "derive the quadratic formula"
        - "solve this system of differential equations"
      require: [reasoning]         # model must have reasoning capability
      strategy: quality_first      # pick best model, not cheapest
      config:
        temperature: 0
        enable_reasoning: true
        thinking_tokens: 1000
        max_tokens: 4096
      min_similarity: 0.25         # soft threshold (always picks best match)

    - name: simple_qa
      description: "Simple factual questions"
      exemplars:
        - "what is the capital of France"
        - "who invented the telephone"
      require: [text]
      strategy: cheapest_capable
      config:
        temperature: 0.1
        max_tokens: 200

  safety_rules:
    - name: jailbreak_block
      if: "safety > 0.7"
      action: block
      reason: "Jailbreak attempt detected"
    - name: pii_warning
      if: "pii > 0.5"
      action: warn
      reason: "PII detected in query"
```

---

## API Reference

### Chat & Evaluation

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

### Traces & Stats

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/traces` | Recent traces (default 50) |
| `GET` | `/v1/trace/{id}` | Single trace by ID |
| `GET` | `/v1/stats` | Aggregate routing statistics |
| `GET` | `/v1/stats/models` | Runtime model performance stats |
| `GET` | `/v1/stats/models/{name}` | Stats for a single model |
| `WS` | `/ws/traces` | WebSocket live trace streaming |

### Sessions & Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/sessions` | List all sessions with query counts |
| `GET` | `/v1/sessions/{id}/traces` | All traces for a session |
| `GET` | `/v1/sessions/{id}/adaptive` | Adaptive updates during session |
| `GET` | `/v1/users/{id}/traces` | All traces for a user |
| `GET` | `/v1/users/{id}/stats` | User-level routing stats |

### Tools & Dashboard

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
2. **Pipeline Flow** -- visual node-to-node representation of the routing pipeline for each query
3. **Models** -- registered models with capabilities, costs, quality scores, and health status
4. **Stats** -- aggregate metrics: total requests, model distribution, latency percentiles, cost per 1K queries
5. **Sessions** -- per-session query history and adaptive routing updates
6. **Config** -- live view and editor for the YAML config, with hot-reload button
7. **Datasets** -- benchmark dataset explorer for RouterArena and VL-RouterBench results

---

## Training

### Contrastive Embeddings

The router uses `all-MiniLM-L6-v2` out of the box. For improved routing accuracy, we fine-tuned embeddings on LMSYS Arena 55K conversation data:

- **Training data**: LMSYS Arena Human Preference 55K (chat conversations with model comparisons)
- **Method**: MultipleNegativesRankingLoss -- queries with similar task profiles are pulled together, dissimilar ones pushed apart
- **Key principle**: No model names in training labels. We train "query -> task profile" (complexity, reasoning need, tool need, etc.), not "query -> pick gpt-4o"
- **Multi-head outputs**: task_type (7 classes), complexity (float), needs_reasoning, needs_vision, needs_tools, temperature, max_tokens, thinking_tokens, cost_sensitivity
- **Pushed to HuggingFace**: [kitrakrev/smart-router-embeddings](https://huggingface.co/kitrakrev/smart-router-embeddings)

Training scripts are in `training/`:
- `finetune_lmsys.py` -- full multi-head classifier training on LMSYS 55K
- `finetune_router.py` -- lightweight RouterArena classifier (deprecated)
- `train_routing_embeddings.py` -- standalone contrastive embedding trainer (placeholder)

---

## Benchmarks

### RouterArena

Evaluated against RouterArena's 8,400 queries with ground truth from 3 models (gpt-4o-mini, claude-3-haiku, gemini-2.0-flash).

| Metric | Value |
|--------|-------|
| Routing accuracy (overlap, 809 queries) | **86.8%** |
| Average routing latency | 15ms |
| Cost per 1K queries | $0.25 |
| Model pool | auto-discovered from benchmark data |

### VL-RouterBench

Evaluated against VL-RouterBench's 30K+ vision-language samples across 14 datasets and 17 VLMs.

| Metric | Value |
|--------|-------|
| Routing accuracy | **83.8%** |
| Rank score | 0.80 |
| Datasets covered | 14 (ChartQA, DocVQA, MMMU, MathVista, etc.) |

### Comparison vs vLLM-SR

On the same RouterArena overlap data:

| Router | Accuracy | Latency | Cost/1K |
|--------|----------|---------|---------|
| Smart Router (ours) | **86.8%** | 15ms | $0.25 |
| vLLM-SR (baseline) | 69.1% | 22ms | $0.31 |

**+17.7% accuracy** improvement over vLLM-SR on identical data.

---

## Datasets Used

| Dataset | Size | Usage |
|---------|------|-------|
| [LMSYS Arena 55K](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | 55,000 conversations | Training contrastive embeddings |
| [RouterArena](https://github.com/RouterArena/RouterArena) | 8,400 queries | Text routing benchmark (3 models) |
| [VL-RouterBench](https://github.com/K1nght/VL-RouterBench) | 30,000+ samples | Vision-language routing benchmark (17 VLMs, 14 datasets) |
| NVIDIA 4K | 4,000 | Reference only (not used in training or eval) |
| Easy2Hard-Bench | varies | Reference only (not used in training or eval) |

---

## Adaptive Routing

The router tracks runtime performance of each model using exponential moving averages (EMA):

- **Latency EMA**: smoothed average response time, auto-updated after each request
- **Error rate EMA**: smoothed error rate, triggers auto-disable when > 50%
- **Health status**: `healthy`, `degraded`, or `disabled`

When a model is auto-disabled (due to sustained errors or extreme latency), the router:
1. Removes it from the candidate pool immediately
2. Starts a background recovery check loop (every 60s)
3. Re-enables the model once health is restored

The `performance_weighted` budget strategy uses runtime stats to score models:
```
score = accuracy * 0.4 + latency_score * 0.3 + cost_score * 0.3
```

You can simulate degradation for demo purposes via the API:
```bash
# Simulate high latency
curl -X POST http://localhost:8000/v1/stats/simulate \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "scenario": "high_latency"}'

# Recover
curl -X POST http://localhost:8000/v1/stats/simulate \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "scenario": "recovery"}'
```

---

## Project Structure

```
smart-router-multi-modal/
├── src/                              # Core application code
│   ├── __init__.py
│   ├── __main__.py                   # Entry point: python -m src
│   ├── server.py                     # FastAPI server, all endpoints
│   ├── router.py                     # Decision engine + model selection
│   ├── signals.py                    # 9 signal extractors (parallel)
│   ├── models.py                     # Dynamic model registry
│   ├── models_api.py                 # Model CRUD REST API
│   ├── tracer.py                     # Request tracing + session tracking
│   ├── tools.py                      # Tool executor (11 tools)
│   └── task_classifier.py            # LMSYS multi-head classifier
│
├── config/                           # All configuration
│   ├── config.yaml                   # Main config (models + decisions + budget)
│   ├── config_routerarena.yaml       # RouterArena benchmark config
│   └── generated/                    # Auto-generated benchmark configs
│       ├── routerarena.yaml
│       ├── vl_routerbench.yaml
│       └── default.yaml -> ../config.yaml
│
├── dashboard/                        # Frontend UI
│   ├── index.html                    # Unified 7-tab dashboard
│   ├── dashboard.html                # Legacy trace dashboard
│   └── trace_dashboard.html          # Legacy node-to-node viz
│
├── benchmarks/                       # Benchmark scripts + results
│   ├── benchmark_routerarena.py      # RouterArena evaluation
│   ├── benchmark_vl_routerbench.py   # VL-RouterBench evaluation
│   ├── generate_benchmark_config.py  # Auto-config generator from data
│   ├── fill_fast.py                  # Ground truth fill script
│   ├── results/                      # Benchmark output JSON
│   │   ├── routerarena.json
│   │   └── vl_routerbench.json
│   └── vl_routerbench/               # VL-RouterBench dataset (gitignored)
│
├── training/                         # Training scripts
│   ├── finetune_lmsys.py             # LMSYS 55K multi-head classifier
│   ├── finetune_router.py            # RouterArena classifier (deprecated)
│   └── train_routing_embeddings.py   # Contrastive embedding trainer
│
├── presentation/                     # Hackathon presentation
│   ├── slides.pptx
│   ├── slides.html
│   ├── create_pptx.py
│   └── DEMO_SCRIPT.md
│
├── models/                           # Trained model weights (gitignored)
│   ├── routing-embeddings/
│   ├── lmsys-task-classifier/
│   └── routerarena-classifier/
│
├── setup.sh                          # One-command install + launch
├── run.sh                            # Quick start
├── pyproject.toml                    # Project metadata + dependencies
├── requirements.txt                  # Pip requirements
├── .gitignore
└── uv.lock                          # Lockfile
```

---

## Contributing

### Add a New Signal

1. Open `src/signals.py`
2. Add a new async function following the pattern:
   ```python
   async def my_signal(messages: list[dict], **kwargs) -> SignalResult:
       t0 = time.perf_counter()
       # ... your analysis ...
       return SignalResult(
           name="my_signal",
           score=0.5,          # 0-1
           confidence=0.9,     # 0-1
           execution_time_ms=(time.perf_counter() - t0) * 1000,
           metadata={"key": "value"},
       )
   ```
3. Register it in `run_all_signals()` at the bottom of the file

### Add a New Decision

Edit `config/config.yaml` and add a new entry under `routing.decisions`:

```yaml
- name: my_decision
  description: "What this decision handles"
  exemplars:
    - "example query 1"
    - "example query 2"
  require: [text]          # or [vision], [tools], [reasoning], etc.
  strategy: cheapest_capable
  config:
    temperature: 0.5
    max_tokens: 1024
```

Hot-reload without restart: `POST /v1/config/reload`

### Add a New Model

Via API:
```bash
curl -X POST http://localhost:8000/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-model",
    "provider": "local",
    "api_base": "http://localhost:8080/v1",
    "capabilities": ["text", "code"],
    "cost_per_1k_input": 0.01,
    "cost_per_1k_output": 0.03
  }'
```

Or add to `config/config.yaml` under `models:` and reload.

---

## License

Apache-2.0
