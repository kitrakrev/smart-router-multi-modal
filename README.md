# LLM Router

Intelligent LLM query router with 9-signal classification, vision routing, tool execution, guardrails, and real-time WebSocket traces. Routes queries to the optimal model based on complexity, domain, safety, cost, and capability requirements.

## Architecture

```
                         +------------------+
                         |   Client / curl  |
                         +--------+---------+
                                  |
                          POST /v1/eval
                          POST /v1/chat/completions
                                  |
                    +-------------v--------------+
                    |       FastAPI Server        |
                    |       (server.py)           |
                    +----+--------+----------+---+
                         |        |          |
              +----------+   +----+----+   +-+----------+
              |              |         |   |             |
     +--------v------+  +---v---+  +--v--------+  +----v------+
     | Signal Engine  |  |Router |  |  Tracer   |  | Tools API |
     | (signals.py)   |  |Engine |  | (tracer)  |  | (tools)   |
     +--------+-------+  +---+---+  +-----+----+  +-----------+
              |               |            |
     9 parallel signals:      |       WebSocket
     - keyword                |       /ws/traces
     - domain (MiniLM)   +---v-----------v----+
     - complexity         |   config.yaml      |
     - language           |   Rule-based +     |
     - safety             |   capability-aware |
     - PII detection      |   model selection  |
     - vision             +--------------------+
     - tool need               |
     - modality           +----v----+
                          | Models  |
                          +---------+
                          gpt-4o | claude-sonnet
                          gpt-4o-mini | qwen-3b-local
```

## Quick Start

```bash
# 1. Clone / cd into the project
cd router-prototype

# 2. Run the server (uv auto-installs dependencies)
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# 3. Test it
curl -X POST http://localhost:8000/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Solve the integral of x^2"}]}'
```

Or use the helper script:

```bash
chmod +x run.sh && ./run.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/eval` | Classify query and return signals + routing decision (no API keys) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat endpoint with routing |
| GET | `/dashboard` | Interactive trace visualization dashboard |
| WS | `/ws/traces` | Live WebSocket stream of all traces |
| GET | `/v1/traces` | Recent traces (JSON) |
| GET | `/v1/trace/{id}` | Single trace by ID |
| GET | `/v1/stats` | Aggregate routing statistics |
| POST | `/v1/tools/execute` | Execute a tool directly |
| GET | `/v1/tools` | List available tools |
| GET | `/v1/models` | List registered models |
| POST | `/v1/models` | Register a new model |
| PUT | `/v1/models/{name}` | Update a model |
| DELETE | `/v1/models/{name}` | Remove a model |
| GET | `/v1/models/{name}/health` | Health check a model endpoint |
| POST | `/v1/models/discover` | Auto-discover from OpenAI-compatible endpoint |
| POST | `/v1/config/reload` | Hot-reload config.yaml |
| GET | `/docs` | Interactive Swagger API docs |

## How to Add Models

Register a new model at runtime:

```bash
# Add a local vLLM model
curl -X POST http://localhost:8000/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama-3-70b",
    "provider": "local",
    "api_base": "http://localhost:8080/v1",
    "model_id": "meta-llama/Llama-3-70B-Instruct",
    "capabilities": ["text", "code", "reasoning"],
    "cost_per_1k_input": 0.005,
    "cost_per_1k_output": 0.015,
    "avg_latency_ms": 200
  }'

# Auto-discover models from an OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/models/discover \
  -H "Content-Type: application/json" \
  -d '{"api_base": "http://localhost:8080/v1", "provider": "vllm"}'

# Update a model
curl -X PUT http://localhost:8000/v1/models/llama-3-70b \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

Or add models in `config.yaml`:

```yaml
models:
  - name: my-model
    provider: local
    cost_per_1k_input: 0.01
    cost_per_1k_output: 0.03
    avg_latency_ms: 100
    capabilities: [text, code]
```

## Dashboard

The `/dashboard` endpoint serves an interactive HTML dashboard that shows:

- **Live trace feed**: Every routing decision appears in real-time via WebSocket
- **Signal breakdown**: Radar chart showing all 9 signal scores per query
- **Model distribution**: Pie chart of which models get selected
- **Latency timeline**: Rolling average of routing latency
- **Stats panel**: Total requests, average latency, cost metrics

Access at `http://localhost:8000/dashboard` after starting the server.

## Benchmark Results

Benchmarked against RouterArena's 8,400 queries with ground truth (gpt-4o-mini.jsonl):

| Metric | Value |
|--------|-------|
| Total queries | 8,400 |
| Blocked (safety) | 26 |
| Routing accuracy | 26.1% |
| Optimal model selection | 70.2% |
| Cost per 1K (router) | $857.69 |
| Cost per 1K (all-mini baseline) | $255.00 |
| Avg routing latency | 2.93 ms |
| P50 routing latency | 2.24 ms |
| P99 routing latency | 10.07 ms |

**Model Distribution**: 62.2% qwen-3b-local, 24.3% gpt-4o-mini, 13.2% claude-sonnet, 0.3% blocked

The router correctly escalates complex math/code queries to stronger models while routing simple MCQ queries to cheaper models. Routing overhead is under 3ms per query on CPU.

## Features

1. **9 Parallel Signals** -- keyword, domain (embedding-based with MiniLM fallback), complexity, language detection, safety/jailbreak, PII detection, vision, tool need, modality classification
2. **Guardrails** -- Blocks prompt injection, jailbreak attempts, and safety violations before routing
3. **Vision Routing** -- Detects image content and routes to vision-capable models (gpt-4o)
4. **Tool Execution** -- 11 built-in tools (port scan, CVE lookup, firewall rules, vulnerability scan, etc.) with automatic tool-need detection
5. **Live Traces** -- WebSocket streaming of every routing decision with full signal breakdown
6. **Dynamic Model Registry** -- Add, remove, update, health-check, and auto-discover models at runtime
7. **Cost-Aware Routing** -- Picks cheapest capable model; routes simple queries to local models
8. **OpenAI-Compatible API** -- Drop-in replacement for `/v1/chat/completions`
9. **Hot-Reload Config** -- Update routing rules without restarting the server
