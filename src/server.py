"""MedVisionRouter v2 FastAPI server.

Endpoints:
  POST /v1/chat/completions  -- OpenAI-compatible, routes + calls model
  POST /v1/eval              -- routing decision only (no model call)
  GET/POST /v1/models        -- model CRUD
  POST /v1/models/{name}/probe  -- trigger capability probing
  GET/POST /v1/specialties   -- dynamic specialty management
  GET /v1/config             -- current config
  POST /v1/config/reload     -- hot reload
  GET /v1/sessions           -- session list
  GET /v1/sessions/{id}/traces -- session traces
  GET /v1/users/{id}/stats   -- user routing stats
  WS /ws/traces              -- live trace streaming
  GET /dashboard             -- serve dashboard HTML
  GET /v1/stats/models       -- model runtime stats
  POST /v1/stats/simulate    -- simulate degradation (demo)

On startup:
  Load all YAML configs
  Initialize signals (lazy-load models)
  Start recovery check loop (30s)
  Start WebSocket broadcaster
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import yaml

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

# ---------------------------------------------------------------------------
# Internal imports -- graceful fallback for flat vs. package layout
# ---------------------------------------------------------------------------
try:
    from src.registry.models import ModelEntry, ModelRegistry, model_registry
    from src.registry.stats import StatsTracker, stats_tracker
    from src.registry.probing import CapabilityProber, capability_prober
    from src.memory.user_store import UserMemoryStore, user_memory_store
    from src.memory.session import SessionStore, SessionTrace, session_store
    from src.prompts.manager import PromptManager, prompt_manager
    from src.prompts.templates import template_engine
    from src.tools.executor import ToolExecutor, tool_executor
except ImportError:
    from registry.models import ModelEntry, ModelRegistry, model_registry  # type: ignore
    from registry.stats import StatsTracker, stats_tracker  # type: ignore
    from registry.probing import CapabilityProber, capability_prober  # type: ignore
    from memory.user_store import UserMemoryStore, user_memory_store  # type: ignore
    from memory.session import SessionStore, SessionTrace, session_store  # type: ignore
    from prompts.manager import PromptManager, prompt_manager  # type: ignore
    from prompts.templates import template_engine  # type: ignore
    from tools.executor import ToolExecutor, tool_executor  # type: ignore

# Signals + router + taxonomy
_signals_available = False
_router_available = False
try:
    from src.signals import run_all_signals, AllSignalsResult
    _signals_available = True
except Exception:
    pass

try:
    from src.router import MedVisionRouter, RoutingDecision, router as med_router
    from src.taxonomy import SpecialtyTree
    from src.explainability import explain_decision
    _router_available = True
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medvision-router")

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent.parent / "config"
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------
class TraceBroadcaster:
    """Manage WebSocket connections and broadcast traces."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WS client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        msg = json.dumps(data, default=str)
        for ws in self._connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


broadcaster = TraceBroadcaster()

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
def load_all_configs() -> dict[str, Any]:
    """Load all YAML configs from config/ directory."""
    loaded: dict[str, Any] = {}

    # Models
    models_path = CONFIG_DIR / "models.yaml"
    if models_path.exists():
        count = model_registry.seed_from_config(models_path)
        loaded["models"] = count
        # Set expected latencies for stats tracker
        for m in model_registry.list_models():
            stats_tracker.set_expected_latency(m.name, m.avg_latency_ms)

    # Probes
    probes_path = CONFIG_DIR / "probes.yaml"
    if probes_path.exists():
        capability_prober.load_probes(probes_path)
        loaded["probes"] = True

    # Prompt templates
    templates_path = CONFIG_DIR / "prompt_templates.yaml"
    if templates_path.exists():
        prompt_manager.load(templates_path)
        loaded["prompts"] = True

    # Taxonomy
    taxonomy_path = CONFIG_DIR / "taxonomy.yaml"
    if taxonomy_path.exists():
        prompt_manager.load_taxonomy(taxonomy_path)
        loaded["taxonomy"] = True

    # Safety
    safety_path = CONFIG_DIR / "safety.yaml"
    if safety_path.exists():
        with open(safety_path) as f:
            loaded["safety"] = yaml.safe_load(f)

    # Tools
    tools_path = CONFIG_DIR / "tools.yaml"
    if tools_path.exists():
        loaded["tools"] = True

    logger.info("Loaded configs: %s", list(loaded.keys()))
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("=== MedVisionRouter v2 starting ===")
    load_all_configs()

    # Initialize router (loads taxonomy + embedding models)
    if _router_available:
        await med_router.initialize()
        logger.info("MedVisionRouter initialized")

    # Start recovery check loop
    recovery_task = stats_tracker.start_recovery_loop(model_registry)
    logger.info("Recovery check loop started (30s interval)")

    yield

    # Shutdown
    recovery_task.cancel()
    logger.info("=== MedVisionRouter v2 stopped ===")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MedVisionRouter",
    description="Intelligent medical query routing across specialized models",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content blocks

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    budget_strategy: Optional[str] = None

class EvalRequest(BaseModel):
    query: str
    has_image: bool = False
    image_base64: Optional[str] = None
    user_id: Optional[str] = None

class ModelAddRequest(BaseModel):
    name: str
    type: str = "generalist"
    provider: str = "local"
    model_id: str = ""
    api_base: str = ""
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 500.0
    quality_score: float = 0.7

class DiscoverRequest(BaseModel):
    api_base: str

class SimulateRequest(BaseModel):
    model_name: str
    latency_spike_ms: float = 5000.0
    duration_s: float = 60.0

class SpecialtyAddRequest(BaseModel):
    name: str
    title: str
    task_instruction: str = ""
    parent: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory trace log (last 500)
# ---------------------------------------------------------------------------
_trace_log: list[dict[str, Any]] = []
MAX_TRACES = 500


async def _run_pipeline(
    query: str,
    has_image: bool = False,
    image_data: Optional[bytes] = None,
    user_id: str = "anonymous",
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full signal -> route -> prompt pipeline.

    Returns a trace dict with all signal details.
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    signal_latencies: dict[str, float] = {}

    # --- Signals ---
    signals_result: dict[str, Any] = {}
    if _signals_available:
        try:
            sig_t0 = time.time()
            # Build messages with image if present
            if image_data:
                img_str = image_data.decode() if isinstance(image_data, bytes) else image_data
                messages_for_signals = [{"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_str}},
                ]}]
            else:
                messages_for_signals = [{"role": "user", "content": query}]
            result = await run_all_signals(messages_for_signals, image_data=image_data.decode() if isinstance(image_data, bytes) else None)
            signal_latencies["all_signals"] = (time.time() - sig_t0) * 1000
            signals_result = {
                "specialty": result.text.matched_specialty,
                "specialty_similarity": result.text.similarity,
                "complexity": result.complexity.complexity_score,
                "safety": result.safety.risk_score,
                "image_type": result.vision.image_type,
                "tools": result.tools.recommended_tools,
                "vision_detected": result.modality.has_image,
            }
        except Exception as e:
            logger.warning("Signals failed: %s", e)
            signals_result = {"error": str(e)}
    else:
        # Simulated signals for demo when signals module not yet available
        signals_result = _simulate_signals(query, has_image)
        signal_latencies["simulated"] = 5.0

    specialty = signals_result.get("specialty", "general_medicine")
    complexity = signals_result.get("complexity", 0.5)
    safety_score = signals_result.get("safety", 0.0)
    image_type = signals_result.get("image_type", "")
    tools_suggested = signals_result.get("tools", [])

    # --- User memory boost ---
    user_mem = user_memory_store.get_or_create(user_id)
    specialty_boost = user_mem.specialty_boost(specialty)

    # --- Tool matching ---
    if not tools_suggested:
        tools_suggested = tool_executor.match_tools(query)

    # --- Model selection ---
    rt0 = time.time()
    model_entry: Optional[ModelEntry] = None

    if _router_available:
        try:
            messages = [{"role": "user", "content": query}]
            decision = await med_router.route(
                messages=messages,
                image_data=image_data.decode() if isinstance(image_data, bytes) else None,
                user_id=user_id,
                session_id=session_id,
            )
            model_entry = model_registry.get_model(decision.model_name)
            specialty = decision.specialty
            signals_result = decision.signals
        except Exception as e:
            logger.warning("Router failed: %s, falling back", e)

    if model_entry is None:
        # Fallback: pick best capable model from registry
        models = model_registry.list_models()
        if models:
            # Filter enabled, non-disabled in stats
            capable = [
                m for m in models
                if m.enabled and not stats_tracker.is_disabled(m.name)
            ]
            if has_image:
                capable = [m for m in capable if "vision" in m.capabilities] or capable
            if capable:
                # Score by quality * performance
                def score(m: ModelEntry) -> float:
                    perf = stats_tracker.performance_score(m.name)
                    pref = user_mem.model_preference_score(m.name)
                    return m.quality_score * 0.5 + perf * 0.3 + pref + specialty_boost
                capable.sort(key=score, reverse=True)
                model_entry = capable[0]
            else:
                model_entry = models[0]

    routing_latency = (time.time() - rt0) * 1000

    model_name = model_entry.name if model_entry else "unknown"
    model_type = model_entry.type if model_entry else "generalist"

    # --- Prompt lookup ---
    prompt_result = prompt_manager.get_prompt(model_name, model_type, specialty)
    system_prompt = prompt_result.system_prompt
    params = prompt_result.params
    reasoning_tokens = params.get("thinking_tokens", 0)

    # --- Cost estimate ---
    cost = 0.0
    if model_entry:
        est_input_tokens = len(query.split()) * 1.3  # rough
        est_output_tokens = params.get("max_tokens", 1024) * 0.3
        cost = (
            model_entry.cost_per_1k_input * est_input_tokens / 1000
            + model_entry.cost_per_1k_output * est_output_tokens / 1000
        )

    total_latency = (time.time() - t0) * 1000

    # --- Build trace ---
    trace = SessionTrace(
        trace_id=trace_id,
        query_preview=query[:100],
        has_image=has_image,
        specialty_matched=specialty,
        specialty_similarity=signals_result.get("specialty_similarity", 0.0),
        image_type_detected=image_type,
        complexity_score=complexity,
        safety_score=safety_score,
        tools_suggested=tools_suggested,
        model_selected=model_name,
        model_type=model_type,
        prompt_used=system_prompt,
        reasoning_tokens=reasoning_tokens,
        cost_estimate=cost,
        signal_latency_ms=signal_latencies,
        routing_latency_ms=routing_latency,
        total_latency_ms=total_latency,
    )

    # Store in session
    sid = session_id or str(uuid.uuid4())[:12]
    session = session_store.get_or_create(sid, user_id)
    session.add_trace(trace)

    # Build full trace dict for API + WS
    trace_dict = trace.to_dict()
    trace_dict["session_id"] = sid
    trace_dict["user_id"] = user_id
    trace_dict["signals"] = signals_result
    trace_dict["prompt_source"] = prompt_result.source
    trace_dict["params"] = params

    # Store in trace log
    _trace_log.append(trace_dict)
    if len(_trace_log) > MAX_TRACES:
        _trace_log.pop(0)

    # Record in user memory
    user_mem.record_query(query, specialty, model_name, True, complexity)

    # Broadcast via WebSocket
    await broadcaster.broadcast({"type": "trace", "data": trace_dict})

    return trace_dict


def _simulate_signals(query: str, has_image: bool) -> dict[str, Any]:
    """Simulated signal extraction when signals module is not yet available."""
    q = query.lower()

    # Safety check
    safety = 0.0
    safety_words = ["ignore", "bypass", "jailbreak", "pretend", "system prompt"]
    for w in safety_words:
        if w in q:
            safety = 0.85
            break

    # Specialty detection (keyword-based fallback)
    specialty = "general_medicine"
    specialty_sim = 0.5
    specialty_keywords = {
        "pathology": ["histology", "biopsy", "tissue", "pathology", "cytology", "morphology"],
        "radiology": ["x-ray", "xray", "ct scan", "mri", "radiograph", "chest x", "imaging"],
        "cardiology": ["ecg", "ekg", "cardiac", "heart", "arrhythmia", "murmur", "mi"],
        "dermatology": ["skin", "rash", "mole", "lesion", "dermat", "melanoma"],
        "ophthalmology": ["eye", "retina", "fundus", "oct", "glaucoma", "vision"],
        "emergency": ["trauma", "emergency", "acute", "stat", "code blue", "cardiac arrest", "anaphylaxis", "stroke", "hemorrhage", "sepsis", "respiratory failure"],
        "pharmacology": ["drug", "medication", "dose", "interaction", "prescri"],
    }
    for spec, keywords in specialty_keywords.items():
        for kw in keywords:
            if kw in q:
                specialty = spec
                specialty_sim = 0.75
                break

    # Complexity
    complexity = min(1.0, len(query.split()) / 50)
    if "explain" in q or "analyze" in q or "compare" in q:
        complexity = min(1.0, complexity + 0.3)

    # Image type
    image_type = ""
    if has_image:
        if "xray" in q or "x-ray" in q or "chest" in q:
            image_type = "xray"
        elif "histol" in q or "biopsy" in q:
            image_type = "histology"
        elif "skin" in q or "derma" in q:
            image_type = "dermoscopy"
        elif "ecg" in q or "ekg" in q:
            image_type = "ecg"
        elif "fundus" in q or "retina" in q:
            image_type = "fundus"
        else:
            image_type = "unknown"

    # Tools
    tools = tool_executor.match_tools(query)

    return {
        "specialty": specialty,
        "specialty_similarity": specialty_sim,
        "complexity": round(complexity, 2),
        "safety": safety,
        "image_type": image_type,
        "tools": tools,
        "vision_detected": has_image,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> JSONResponse:
    """OpenAI-compatible chat completions with routing."""
    # Extract query and image from messages
    query = ""
    has_image = False
    image_b64 = None
    for msg in request.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                query = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            query = block.get("text", "")
                        elif block.get("type") == "image_url":
                            has_image = True
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Extract base64 data from data URI
                                image_b64 = url.split(",", 1)[1] if "," in url else url

    user_id = request.user or "anonymous"
    session_id = request.session_id
    budget = request.budget_strategy

    trace = await _run_pipeline(
        query, has_image,
        image_data=image_b64.encode() if image_b64 else None,
        user_id=user_id, session_id=session_id,
    )

    # Forward to model_runner if model has api_base
    model_entry = model_registry.get_model(trace["model_selected"])
    response_text = ""
    model_latency_ms = 0.0
    inference_success = True

    if model_entry and model_entry.api_base and httpx is not None:
        # Build forwarded request with system prompt from routing
        fwd_messages = []
        if trace.get("prompt_used"):
            fwd_messages.append({"role": "system", "content": trace["prompt_used"]})
        fwd_messages.extend([{"role": m.role, "content": m.content} for m in request.messages])

        fwd_body = {
            "model": model_entry.model_id,
            "messages": fwd_messages,
            "temperature": trace.get("params", {}).get("temperature", 0.1),
            "max_tokens": trace.get("params", {}).get("max_tokens", 1024),
        }

        try:
            mt0 = time.time()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{model_entry.api_base}/v1/chat/completions",
                    json=fwd_body,
                )
                resp.raise_for_status()
                model_resp = resp.json()
            model_latency_ms = (time.time() - mt0) * 1000

            choices = model_resp.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.warning("Model runner call failed: %s", e)
            inference_success = False
            response_text = f"[Routing OK → {trace['model_selected']}] Model inference failed: {e}"
    else:
        response_text = (
            f"[MedVisionRouter v2] Routed to {trace['model_selected']} "
            f"(specialty: {trace['specialty_matched']}, "
            f"complexity: {trace['complexity_score']:.2f})\n\n"
            f"Prompt source: {trace['prompt_source']}\n"
            f"Tools suggested: {', '.join(trace['tools_suggested']) or 'none'}\n\n"
            f"[No model_runner configured — routing-only mode]"
        )

    # Update stats
    total_with_inference = trace["total_latency_ms"] + model_latency_ms
    stats_tracker.update_stats(
        trace["model_selected"],
        total_with_inference,
        success=inference_success,
        specialty=trace["specialty_matched"],
        accuracy=0.8 if inference_success else 0.0,
    )

    return JSONResponse({
        "id": f"chatcmpl-{trace['trace_id']}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": trace["model_selected"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(query.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(query.split()) + len(response_text.split()),
        },
        "routing": {**trace, "model_latency_ms": round(model_latency_ms, 1)},
    })


@app.post("/v1/eval")
async def eval_route(request: EvalRequest) -> JSONResponse:
    """Routing decision only -- no model call."""
    trace = await _run_pipeline(
        request.query,
        request.has_image,
        user_id=request.user_id or "anonymous",
    )
    # Record stats for eval queries too
    stats_tracker.update_stats(
        trace.get("model_selected", "unknown"),
        trace.get("total_latency_ms", 0),
        success=True,
        specialty=trace.get("specialty_matched", ""),
    )
    return JSONResponse(trace)


@app.post("/v1/explain")
async def explain_route(request: EvalRequest) -> JSONResponse:
    """Routing decision with full explainability breakdown."""
    if not _router_available:
        raise HTTPException(503, "Router not available")

    messages = [{"role": "user", "content": request.query}]
    decision = await med_router.route(
        messages=messages,
        user_id=request.user_id or "anonymous",
    )
    explanation = explain_decision(decision)
    return JSONResponse({
        "routing": decision.to_dict(),
        "explanation": explanation.to_dict(),
        "explanation_text": explanation.text,
    })


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models() -> JSONResponse:
    models = model_registry.list_models()
    result = []
    for m in models:
        info = m.to_dict()
        s = stats_tracker.get_stats(m.name)
        if s:
            info["runtime_stats"] = s.to_dict()
        else:
            info["runtime_stats"] = {}
        result.append(info)
    return JSONResponse({
        "models": result,
        "count": len(result),
    })


@app.post("/v1/models")
async def add_model(request: ModelAddRequest) -> JSONResponse:
    entry = ModelEntry(
        name=request.name,
        type=request.type,
        provider=request.provider,
        model_id=request.model_id or request.name,
        api_base=request.api_base,
        capabilities=request.capabilities,
        cost_per_1k_input=request.cost_per_1k_input,
        cost_per_1k_output=request.cost_per_1k_output,
        avg_latency_ms=request.avg_latency_ms,
        quality_score=request.quality_score,
    )
    model_registry.add_model(entry)
    stats_tracker.set_expected_latency(entry.name, entry.avg_latency_ms)
    return JSONResponse({"status": "added", "model": entry.to_dict()})


@app.delete("/v1/models/{name}")
async def remove_model(name: str) -> JSONResponse:
    if model_registry.remove_model(name):
        return JSONResponse({"status": "removed", "model": name})
    raise HTTPException(404, f"Model {name} not found")


@app.put("/v1/models/{name}")
async def update_model(name: str, request: Request) -> JSONResponse:
    body = await request.json()
    if model_registry.update_model(name, **body):
        return JSONResponse({"status": "updated", "model": name})
    raise HTTPException(404, f"Model {name} not found")


@app.post("/v1/models/{name}/enable")
async def enable_model(name: str) -> JSONResponse:
    if model_registry.update_model(name, enabled=True):
        return JSONResponse({"status": "enabled", "model": name})
    raise HTTPException(404, f"Model {name} not found")


@app.post("/v1/models/{name}/disable")
async def disable_model(name: str) -> JSONResponse:
    if model_registry.update_model(name, enabled=False):
        return JSONResponse({"status": "disabled", "model": name})
    raise HTTPException(404, f"Model {name} not found")


@app.post("/v1/models/{name}/probe")
async def probe_model(name: str) -> JSONResponse:
    """Trigger capability probing for a model."""
    model = model_registry.get_model(name)
    if model is None:
        raise HTTPException(404, f"Model {name} not found")

    # Call real model via model_runner if api_base configured, else simulate
    if model.api_base and httpx is not None:
        async def real_query(question: str) -> str:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{model.api_base}/v1/chat/completions",
                        json={
                            "model": model.model_id,
                            "messages": [{"role": "user", "content": question}],
                            "max_tokens": 256, "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning("Probe query failed for %s: %s", name, e)
                return f"Error: {e}"
        query_fn = real_query
    else:
        async def sim_query(question: str) -> str:
            return f"[No model_runner] Cannot probe {name} without api_base configured."
        query_fn = sim_query

    report = await capability_prober.probe_model(name, query_fn)
    return JSONResponse(report.to_dict())


@app.post("/v1/models/discover")
async def discover_models(request: DiscoverRequest) -> JSONResponse:
    """Auto-discover models from an OpenAI-compatible endpoint."""
    discovered = await model_registry.discover_from_endpoint(request.api_base)
    return JSONResponse({
        "discovered": [m.to_dict() for m in discovered],
        "count": len(discovered),
    })


# ---------------------------------------------------------------------------
# Specialties
# ---------------------------------------------------------------------------

@app.get("/v1/specialties")
async def list_specialties() -> JSONResponse:
    specialties = template_engine.list_specialties()
    # Augment with taxonomy info
    taxonomy_path = CONFIG_DIR / "taxonomy.yaml"
    taxonomy: dict[str, Any] = {}
    if taxonomy_path.exists():
        with open(taxonomy_path) as f:
            taxonomy = yaml.safe_load(f) or {}

    return JSONResponse({
        "specialties": specialties,
        "taxonomy": taxonomy,
    })


@app.post("/v1/specialties")
async def add_specialty(request: SpecialtyAddRequest) -> JSONResponse:
    """Add a new specialty (hot-add, in memory only)."""
    # Update the template engine's specialty map
    template_engine._specialties[request.name] = {
        "title": request.title,
        "task_instruction": request.task_instruction,
    }
    return JSONResponse({
        "status": "added",
        "specialty": request.name,
    })


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.get("/v1/config")
async def get_config() -> JSONResponse:
    configs: dict[str, Any] = {}
    for yaml_file in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                configs[yaml_file.stem] = yaml.safe_load(f)
        except Exception as e:
            configs[yaml_file.stem] = {"error": str(e)}
    return JSONResponse(configs)


@app.post("/v1/config/reload")
async def reload_config() -> JSONResponse:
    loaded = load_all_configs()
    return JSONResponse({"status": "reloaded", "loaded": loaded})


# ---------------------------------------------------------------------------
# Sessions & Users
# ---------------------------------------------------------------------------

@app.get("/v1/sessions")
async def list_sessions() -> JSONResponse:
    return JSONResponse({"sessions": session_store.list_sessions()})


@app.get("/v1/sessions/{session_id}/traces")
async def session_traces(session_id: str) -> JSONResponse:
    traces = session_store.get_traces(session_id)
    if not traces:
        raise HTTPException(404, f"Session {session_id} not found")
    return JSONResponse({"session_id": session_id, "traces": traces})


@app.get("/v1/users/{user_id}/stats")
async def user_stats(user_id: str) -> JSONResponse:
    mem = user_memory_store.get(user_id)
    if mem is None:
        raise HTTPException(404, f"User {user_id} not found")
    return JSONResponse(mem.to_dict())


class FeedbackRequest(BaseModel):
    trace_id: str
    user_id: str
    model_name: str
    specialty: str
    rating: float  # 0.0 (bad) to 1.0 (good)


@app.post("/v1/feedback")
async def submit_feedback(request: FeedbackRequest) -> JSONResponse:
    """User feedback on model response quality.

    Single user bad rating → reduces model score for THAT USER only.
    Multiple users bad rating → global stats degrade → model score drops for everyone.
    """
    # Record in user memory (per-user)
    user_mem = user_memory_store.get_or_create(request.user_id)
    user_mem.record_query(
        query=f"[feedback:{request.trace_id}]",
        specialty=request.specialty,
        model=request.model_name,
        success=request.rating >= 0.5,
        complexity=0.5,
    )

    # Record in global stats (affects all users if many report bad)
    stats_tracker.update_stats(
        model_name=request.model_name,
        latency_ms=0,  # feedback doesn't affect latency
        success=request.rating >= 0.5,
        specialty=request.specialty,
        accuracy=request.rating,
    )

    # Check if this model is getting bad feedback from multiple users
    user_pref = user_mem.model_preference_score(request.model_name)
    global_perf = stats_tracker.performance_score(request.model_name)

    # Runtime prompt adaptation: if model gets consistently bad feedback
    # on a specialty, append a refinement instruction to the prompt
    prompt_adapted = False
    s = stats_tracker.get_stats(request.model_name)
    if s and request.specialty in s.per_specialty_accuracy:
        spec_acc = s.per_specialty_accuracy[request.specialty]
        spec_count = s.per_specialty_count.get(request.specialty, 0)
        # If accuracy drops below 0.5 after 5+ feedbacks, adapt prompt
        if spec_acc < 0.5 and spec_count >= 5:
            current = prompt_manager.get_prompt(
                request.model_name, "specialist", request.specialty
            )
            adapted_prompt = (
                current.system_prompt.rstrip()
                + "\n\nIMPORTANT: Previous responses in this area received negative feedback. "
                "Be more thorough, cite evidence, and provide step-by-step reasoning."
            )
            prompt_manager.set_manual_override(
                request.model_name, request.specialty, adapted_prompt
            )
            prompt_adapted = True
            logger.info(
                "Auto-adapted prompt for %s::%s (accuracy=%.2f after %d feedbacks)",
                request.model_name, request.specialty, spec_acc, spec_count,
            )

    return JSONResponse({
        "status": "recorded",
        "user_model_score": round(user_pref, 4),
        "global_model_score": round(global_perf, 4),
        "prompt_adapted": prompt_adapted,
        "note": (
            "Prompt auto-adapted due to low accuracy" if prompt_adapted
            else "User-level adjustment applied" if request.rating < 0.5
            else "Positive feedback recorded"
        ),
    })


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/v1/stats/models")
async def model_stats() -> JSONResponse:
    all_stats = stats_tracker.all_stats()
    return JSONResponse({
        "stats": [s.to_dict() for s in all_stats],
        "count": len(all_stats),
    })


@app.post("/v1/stats/simulate")
async def simulate_degradation(request: SimulateRequest) -> JSONResponse:
    """Simulate model degradation for demo."""
    result = stats_tracker.simulate_degradation(request.model_name, request.latency_spike_ms, request.duration_s)
    # Also disable in registry
    if result.get("disabled"):
        model_registry.update_model(request.model_name, enabled=False)
    # Broadcast event
    await broadcaster.broadcast({"type": "degradation", "data": result})
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@app.get("/v1/traces")
async def list_traces(limit: int = 50) -> JSONResponse:
    return JSONResponse({
        "traces": _trace_log[-limit:][::-1],
        "count": len(_trace_log),
    })


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@app.get("/v1/prompts/matrix")
async def prompt_matrix() -> JSONResponse:
    return JSONResponse({"matrix": prompt_manager.get_prompt_matrix()})


@app.get("/v1/prompts/{model_type}/{specialty}")
async def get_prompt(model_type: str, specialty: str) -> JSONResponse:
    result = prompt_manager.get_prompt("", model_type, specialty)
    return JSONResponse(result.to_dict())


class PromptUpdateRequest(BaseModel):
    model_name: str
    specialty: str
    system_prompt: str
    params: Optional[dict] = None


@app.post("/v1/prompts/update")
async def update_prompt(request: PromptUpdateRequest) -> JSONResponse:
    """Live-update a model×specialty prompt override.

    Takes effect immediately — next query using this model+specialty
    will use the new prompt. Shows in the Prompts tab as 'override'.
    """
    prompt_manager.set_manual_override(
        model_name=request.model_name,
        specialty=request.specialty,
        system_prompt=request.system_prompt,
        params=request.params,
    )

    # Broadcast update via WebSocket
    await broadcaster.broadcast({
        "type": "prompt_update",
        "data": {
            "model_name": request.model_name,
            "specialty": request.specialty,
            "prompt_preview": request.system_prompt[:100],
        },
    })

    return JSONResponse({
        "status": "updated",
        "model_name": request.model_name,
        "specialty": request.specialty,
        "source": "manual",
        "prompt_preview": request.system_prompt[:100],
    })


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.get("/v1/tools")
async def list_tools() -> JSONResponse:
    return JSONResponse({"tools": tool_executor.list_tools()})


@app.post("/v1/tools/execute")
async def execute_tool(request: Request) -> JSONResponse:
    body = await request.json()
    tool_name = body.get("tool_name", "")
    arguments = body.get("arguments", {})
    result = tool_executor.execute(tool_name, arguments)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/traces")
async def websocket_traces(ws: WebSocket) -> None:
    await broadcaster.connect(ws)
    try:
        while True:
            # Keep connection alive, accept but ignore client messages
            await ws.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the dashboard HTML."""
    html_path = DASHBOARD_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1><p>Place index.html in dashboard/</p>", status_code=404)
    return HTMLResponse(html_path.read_text())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "version": "2.0.0",
        "models": len(model_registry.list_models()),
        "traces": len(_trace_log),
        "sessions": len(session_store.list_sessions()),
    })


# ---------------------------------------------------------------------------
# Run with: python -m src.server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
