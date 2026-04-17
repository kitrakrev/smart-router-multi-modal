"""Main FastAPI server for the LLM Router.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible, routes to best model
  POST /v1/eval              — classify query, return signals + routing decision (no API key needed)
  GET  /v1/trace/{id}        — get trace for a request
  GET  /v1/traces            — get recent traces
  GET  /v1/stats             — aggregate statistics
  WS   /ws/traces            — WebSocket for live trace streaming
  GET  /dashboard             — Serve trace visualization dashboard
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from signals import run_all_signals, SignalResult
from router import Router, RouterDecision
from tracer import trace_store, RequestTrace
from tools import ToolExecutor
from models_api import models_router
from models import model_registry

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Router",
    description="Intelligent query routing across multiple LLM providers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router_engine = Router(config_path=str(Path(__file__).parent / "config.yaml"))
router_engine.set_registry(model_registry)
trace_store.set_registry(model_registry)
app.include_router(models_router)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content blocks

class ChatRequest(BaseModel):
    model: Optional[str] = None  # ignored — router decides
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[dict]] = None
    stream: bool = False

class EvalRequest(BaseModel):
    messages: list[ChatMessage]
    tools: Optional[list[dict]] = None

class ChatChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage = ChatUsage()
    routing: Optional[dict] = None


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t_start = time.perf_counter()
    trace_id = trace_store.new_trace_id()

    messages_raw = [m.model_dump() for m in req.messages]
    tools_raw = req.tools

    # Run all signals
    signals = await run_all_signals(messages_raw, tools=tools_raw)

    # Route (pass query text for embedding-based matching)
    query_text = _query_preview(messages_raw)
    decision = router_engine.decide(signals, trace_id=trace_id, query_text=query_text)

    # If blocked
    if decision.blocked:
        t_total = (time.perf_counter() - t_start) * 1000
        await trace_store.add_trace(
            trace_id=trace_id,
            query=_query_preview(messages_raw),
            signals=signals,
            decision=decision,
            total_latency_ms=t_total,
        )
        return JSONResponse(status_code=400, content={
            "error": {
                "message": decision.block_reason,
                "type": "content_policy_violation",
                "code": "blocked_by_router",
                "trace_id": trace_id,
            }
        })

    # Simulate model response (in production, forward to real provider)
    t_model_start = time.perf_counter()
    simulated_response = _simulate_model_response(decision, messages_raw)
    model_latency = (time.perf_counter() - t_model_start) * 1000

    t_total = (time.perf_counter() - t_start) * 1000

    # Update runtime stats for the selected model
    domain = "general"
    for s in signals:
        if s.name == "domain":
            domain = s.metadata.get("domain", "general")
            break
    model_registry.update_stats(decision.selected_model, model_latency, True, domain=domain)

    # Store trace
    await trace_store.add_trace(
        trace_id=trace_id,
        query=_query_preview(messages_raw),
        signals=signals,
        decision=decision,
        total_latency_ms=t_total,
        model_response_latency_ms=model_latency,
    )

    # Build OpenAI-compatible response
    return ChatResponse(
        id=f"chatcmpl-{trace_id[:12]}",
        created=int(time.time()),
        model=decision.selected_model,
        choices=[ChatChoice(
            message={"role": "assistant", "content": simulated_response},
        )],
        usage=ChatUsage(prompt_tokens=_estimate_tokens(messages_raw), completion_tokens=50, total_tokens=0),
        routing={
            "trace_id": trace_id,
            "reason": decision.reason,
            "estimated_cost": decision.estimated_cost,
            "decision_name": decision.decision_name,
            "similarity": decision.similarity,
            "signals_summary": {s.name: round(s.score, 3) for s in signals},
        },
    )


# ---------------------------------------------------------------------------
# POST /v1/eval  — No API keys needed
# ---------------------------------------------------------------------------

@app.post("/v1/eval")
async def eval_query(req: EvalRequest):
    t_start = time.perf_counter()
    trace_id = trace_store.new_trace_id()

    messages_raw = [m.model_dump() for m in req.messages]
    tools_raw = req.tools

    # Run all signals
    signals = await run_all_signals(messages_raw, tools=tools_raw)

    # Route (pass query text for embedding-based matching)
    query_text = _query_preview(messages_raw)
    decision = router_engine.decide(signals, trace_id=trace_id, query_text=query_text)

    t_total = (time.perf_counter() - t_start) * 1000

    # Store trace
    await trace_store.add_trace(
        trace_id=trace_id,
        query=_query_preview(messages_raw),
        signals=signals,
        decision=decision,
        total_latency_ms=t_total,
    )

    # Build detailed response
    signal_details = []
    for s in signals:
        signal_details.append({
            "name": s.name,
            "score": round(s.score, 4),
            "confidence": round(s.confidence, 3),
            "execution_time_ms": s.execution_time_ms,
            "metadata": s.metadata,
            "skipped": s.skipped,
        })

    return {
        "trace_id": trace_id,
        "signals": signal_details,
        "decision": {
            "selected_model": decision.selected_model,
            "inference_config": decision.inference_config,
            "estimated_cost": decision.estimated_cost,
            "estimated_latency_ms": decision.estimated_latency_ms,
            "reason": decision.reason,
            "blocked": decision.blocked,
            "decision_name": decision.decision_name,
            "similarity": decision.similarity,
        },
        "total_latency_ms": round(t_total, 2),
        "available_tools": ToolExecutor.available_tools(),
    }


# ---------------------------------------------------------------------------
# GET /v1/trace/{trace_id}
# ---------------------------------------------------------------------------

@app.get("/v1/trace/{trace_id}")
async def get_trace(trace_id: str):
    trace = trace_store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.to_dict()


# ---------------------------------------------------------------------------
# GET /v1/traces  — recent traces
# ---------------------------------------------------------------------------

@app.get("/v1/traces")
async def get_traces(n: int = 50):
    return trace_store.get_recent(n)


# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------

@app.get("/v1/stats")
async def get_stats():
    return trace_store.get_stats()


# ---------------------------------------------------------------------------
# WS /ws/traces  — live trace streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/traces")
async def ws_traces(ws: WebSocket):
    await ws.accept()
    await trace_store.subscribe(ws)
    # Send recent traces as initial payload
    try:
        recent = trace_store.get_recent(20)
        await ws.send_text(json.dumps({"type": "init", "traces": recent, "stats": trace_store.get_stats(), "model_stats": model_registry.get_stats_snapshot()}))
        # Keep alive
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Client can send ping
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await trace_store.unsubscribe(ws)


# ---------------------------------------------------------------------------
# GET /dashboard — Serve HTML dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


# ---------------------------------------------------------------------------
# GET / — Redirect to dashboard
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return HTMLResponse(
        content='<html><head><meta http-equiv="refresh" content="0;url=/dashboard"></head></html>',
        status_code=200,
    )


# ---------------------------------------------------------------------------
# POST /v1/tools/execute — Execute a tool directly
# ---------------------------------------------------------------------------

class ToolExecRequest(BaseModel):
    tool_name: str
    arguments: dict = {}

@app.post("/v1/tools/execute")
async def execute_tool(req: ToolExecRequest):
    result = ToolExecutor.execute(req.tool_name, req.arguments)
    return {
        "tool_name": result.tool_name,
        "success": result.success,
        "output": result.output,
        "execution_time_ms": result.execution_time_ms,
    }


# ---------------------------------------------------------------------------
# GET /v1/tools — List available tools
# ---------------------------------------------------------------------------

@app.get("/v1/tools")
async def list_tools():
    return {"tools": ToolExecutor.get_definitions()}


# ---------------------------------------------------------------------------
# Config API — Dynamic config management (hot reload)
# ---------------------------------------------------------------------------

class BudgetUpdate(BaseModel):
    max_cost_per_query: Optional[float] = None
    strategy: Optional[str] = None
    quality_threshold: Optional[float] = None

class RulesUpdate(BaseModel):
    rules: list[dict]

@app.get("/v1/config")
async def get_config():
    """GET /v1/config -- return the current in-memory configuration."""
    return router_engine.get_config()

@app.put("/v1/config")
async def put_config(request: Request):
    """PUT /v1/config -- replace the full config (hot reload, no restart)."""
    body = await request.json()
    router_engine.apply_config(body)
    return {
        "status": "updated",
        "models": list(router_engine.models.keys()),
        "rules_count": len(router_engine.rules),
        "budget": {
            "max_cost_per_query": router_engine.budget.max_cost_per_query,
            "strategy": router_engine.budget.strategy,
            "quality_threshold": router_engine.budget.quality_threshold,
        },
    }

@app.patch("/v1/config/rules")
async def patch_config_rules(req: RulesUpdate):
    """PATCH /v1/config/rules -- update just routing rules."""
    router_engine.update_rules(req.rules)
    return {
        "status": "rules_updated",
        "rules_count": len(router_engine.rules),
    }

@app.patch("/v1/config/budget")
async def patch_config_budget(req: BudgetUpdate):
    """PATCH /v1/config/budget -- update budget constraints."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    router_engine.update_budget(updates)
    return {
        "status": "budget_updated",
        "budget": {
            "max_cost_per_query": router_engine.budget.max_cost_per_query,
            "strategy": router_engine.budget.strategy,
            "quality_threshold": router_engine.budget.quality_threshold,
        },
    }

@app.post("/v1/config/reload")
async def reload_config():
    """POST /v1/config/reload -- reload from config.yaml file on disk."""
    router_engine.load_config()
    return {
        "status": "reloaded",
        "models": list(router_engine.models.keys()),
        "rules_count": len(router_engine.rules),
        "budget": {
            "max_cost_per_query": router_engine.budget.max_cost_per_query,
            "strategy": router_engine.budget.strategy,
            "quality_threshold": router_engine.budget.quality_threshold,
        },
    }


# ---------------------------------------------------------------------------
# GET /v1/stats/models — Runtime model stats
# ---------------------------------------------------------------------------

@app.get("/v1/stats/models")
async def get_model_stats():
    """Return runtime performance stats for all models."""
    return model_registry.get_stats_snapshot()


@app.get("/v1/stats/models/{name}")
async def get_single_model_stats(name: str):
    """Return runtime stats + latency histogram for a single model."""
    snapshot = model_registry.get_stats_snapshot()
    if name not in snapshot:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found in stats")
    return snapshot[name]


# ---------------------------------------------------------------------------
# POST /v1/stats/simulate — Simulate model degradation (DEMO ONLY)
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    model: str
    scenario: str  # "high_latency" | "errors" | "recovery"

@app.post("/v1/stats/simulate")
async def simulate_degradation(req: SimulateRequest):
    """Simulate a model going slow/down for demo purposes."""
    model_name = req.model
    if model_name not in model_registry.runtime_stats:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if req.scenario == "high_latency":
        for _ in range(20):
            model_registry.update_stats(model_name, 5000, True)
        stats = model_registry.runtime_stats[model_name]
        return {
            "message": f"{model_name} now showing high latency (EMA: {stats.latency_ema:.0f}ms), status: {stats.status}",
            "status": stats.status,
        }

    elif req.scenario == "errors":
        for _ in range(20):
            model_registry.update_stats(model_name, 200, False)
        stats = model_registry.runtime_stats[model_name]
        return {
            "message": f"{model_name} now showing errors (rate: {stats.error_rate_ema:.1%}), status: {stats.status}",
            "status": stats.status,
        }

    elif req.scenario == "recovery":
        stats = model_registry.runtime_stats[model_name]
        model_entry = await model_registry.get_model(model_name)
        if model_entry:
            stats.latency_ema = model_entry.avg_latency_ms
        else:
            stats.latency_ema = 500.0
        stats.error_rate_ema = 0.01
        stats.status = "healthy"
        stats.disabled_reason = ""
        stats.disabled_at = None
        if model_entry:
            model_entry.enabled = True
        return {
            "message": f"{model_name} recovered",
            "status": "healthy",
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'. Use: high_latency, errors, recovery")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_preview(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content[:100]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")[:100]
    return "[no user message]"


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content.split()) * 1.3
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(block.get("text", "").split()) * 1.3
    return int(total)


def _simulate_model_response(decision: RouterDecision, messages: list[dict]) -> str:
    """Generate a simulated response based on the routing decision."""
    model = decision.selected_model
    query = _query_preview(messages)

    # Check if tool routing was used
    tool_needed = False
    for sig_name, sig_data in decision.signals_used.items():
        if sig_name == "tool":
            tool_needed = sig_data.get("metadata", {}).get("needed", False)

    domain = "general"
    for sig_name, sig_data in decision.signals_used.items():
        if sig_name == "domain":
            domain = sig_data.get("metadata", {}).get("domain", "general")
            break

    if tool_needed:
        domain = "tool"

    responses = {
        "tool": f"[{model}] I'll help with that security task. Let me execute the relevant tools...\n\nRunning port scan and vulnerability assessment on the target.\n\n**Results:**\n- Port scan completed: 5 open ports found (22, 80, 443, 8080, 3306)\n- Vulnerability scan: 2 findings (1 high, 1 medium)\n- Reputation check: Score 45/100 (suspicious)\n\nRecommendation: Consider blocking suspicious traffic and patching identified vulnerabilities.",
        "math": f"[{model}] To solve this mathematical problem, I'll work through it step by step using rigorous mathematical reasoning...\n\nLet me analyze the given expression and apply the relevant theorems.\n\nThe solution involves applying the fundamental theorem of calculus...\n\n**Result**: The answer is derived through systematic mathematical analysis.",
        "code": f"[{model}] Here's the implementation:\n\n```python\ndef solution():\n    # Optimized implementation\n    # Time complexity: O(n log n)\n    # Space complexity: O(n)\n    pass\n```\n\nThis solution handles edge cases and follows best practices.",
        "science": f"[{model}] This is a fascinating scientific question. The underlying mechanism involves quantum mechanical principles and thermodynamic equilibrium...\n\nThe key insight is that the phenomenon emerges from the interplay of multiple physical forces.",
        "creative": f"[{model}] *In twilight's gentle, fading glow,*\n*Where autumn leaves dance soft and slow,*\n*A story whispers through the trees,*\n*Of memories carried on the breeze...*",
        "medical": f"[{model}] Based on current medical literature, this condition involves several factors. Please note: this is for informational purposes only and not medical advice. Consult a healthcare professional for diagnosis and treatment.",
        "legal": f"[{model}] From a legal perspective, this matter involves several key considerations under the relevant jurisdiction's framework. Note: this is general legal information, not legal advice.",
        "general": f"[{model}] Great question! Here's a comprehensive answer based on the available information...\n\nThe key points to consider are:\n1. Context and background\n2. Main factors\n3. Practical implications\n\nI hope this helps clarify things!",
    }

    return responses.get(domain, responses["general"])


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    # Start background recovery loop for disabled models
    asyncio.create_task(model_registry.recovery_check_loop())

    print("\n" + "=" * 60)
    print("  LLM Router v2.0 — Embedding-Based Decision Matching + Adaptive Routing")
    print("=" * 60)
    print(f"  Models loaded: {list(router_engine.models.keys())}")
    print(f"  Decisions:     {len(router_engine.decisions)} embedding-based")
    print(f"  Legacy rules:  {len(router_engine.rules)}")
    print(f"  Safety rules:  {len(router_engine.safety_rules)}")
    print(f"  Budget strategy: {router_engine.budget.strategy}")
    print(f"  Max cost/query:  ${router_engine.budget.max_cost_per_query}")
    dm = router_engine.decision_matcher
    print(f"  Matcher ready: {dm.is_loaded} ({len(dm.decision_embeddings)} decisions)")
    print(f"  Dashboard:     http://localhost:8000/dashboard")
    print(f"  API docs:      http://localhost:8000/docs")
    print(f"  Config API:    GET/PUT /v1/config, PATCH /v1/config/rules|budget")
    print("=" * 60 + "\n")
