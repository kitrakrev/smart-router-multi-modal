"""Unit tests verifying every frontend-to-API flow in the MedVisionRouter dashboard.

Each test corresponds to a fetch() call in dashboard/index.html and validates
that the backend returns the response shape the JS code expects.
"""

from __future__ import annotations

import pytest

from tests.conftest import _seed_test_model

pytestmark = pytest.mark.asyncio


# ── 1. Health endpoint ──────────────────────────────────────────────────────

async def test_health(client):
    """GET /health returns status, models count, traces, sessions."""
    _seed_test_model("health-m1")
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert isinstance(data["models"], int)
    assert data["models"] >= 1
    assert "traces" in data
    assert "sessions" in data
    assert "version" in data


# ── 2. Eval endpoint ────────────────────────────────────────────────────────

async def test_eval_basic(client):
    """POST /v1/eval with a text query returns specialty, model, and signals."""
    _seed_test_model("eval-m1")
    resp = await client.post("/v1/eval", json={
        "query": "What are the histological features of adenocarcinoma?",
        "has_image": False,
        "user_id": "test-user",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Fields the dashboard JS reads from the trace
    assert "specialty_matched" in data
    assert "model_selected" in data
    assert "signals" in data
    assert "complexity_score" in data
    assert "safety_score" in data
    assert "trace_id" in data
    assert "total_latency_ms" in data
    assert "cost_estimate" in data
    assert "prompt_used" in data
    assert "prompt_source" in data


async def test_eval_with_image(client):
    """POST /v1/eval with has_image=True returns image type in signals."""
    _seed_test_model("eval-img-m1")
    resp = await client.post("/v1/eval", json={
        "query": "Analyze this chest x-ray",
        "has_image": True,
        "image_base64": "demo",
        "user_id": "test-user",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "image_type_detected" in data


# ── 3. Explain endpoint ─────────────────────────────────────────────────────

async def test_explain_requires_router(client):
    """POST /v1/explain returns 503 when router module is not available."""
    # In test context, the real router (with embedding models) is likely
    # unavailable, so we expect a 503 or a valid response.
    resp = await client.post("/v1/explain", json={
        "query": "What is adenocarcinoma?",
    })
    # Either 503 (router not available) or 200 (router available)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "explanation_text" in data
        assert "routing" in data


# ── 4. Models CRUD ──────────────────────────────────────────────────────────

async def test_list_models(client):
    """GET /v1/models returns models list and count."""
    _seed_test_model("list-m1")
    _seed_test_model("list-m2")
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "count" in data
    assert data["count"] >= 2
    # Each model should have runtime_stats (possibly empty dict)
    for m in data["models"]:
        assert "name" in m
        assert "runtime_stats" in m


async def test_add_model(client):
    """POST /v1/models adds a new model."""
    resp = await client.post("/v1/models", json={
        "name": "new-test-model",
        "type": "specialist",
        "provider": "openai",
        "model_id": "gpt-custom",
        "capabilities": ["text", "vision"],
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "quality_score": 0.9,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "added"
    assert data["model"]["name"] == "new-test-model"

    # Verify it appears in list
    resp2 = await client.get("/v1/models")
    names = [m["name"] for m in resp2.json()["models"]]
    assert "new-test-model" in names


async def test_update_model(client):
    """PUT /v1/models/{name} updates model fields."""
    _seed_test_model("upd-m1")
    resp = await client.put("/v1/models/upd-m1", json={
        "quality_score": 0.95,
        "avg_latency_ms": 100.0,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


async def test_update_model_not_found(client):
    """PUT /v1/models/{name} returns 404 for unknown model."""
    resp = await client.put("/v1/models/nonexistent-xyz", json={"quality_score": 0.5})
    assert resp.status_code == 404


async def test_delete_model(client):
    """DELETE /v1/models/{name} removes a model."""
    _seed_test_model("del-m1")
    resp = await client.delete("/v1/models/del-m1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"

    # Verify gone
    resp2 = await client.get("/v1/models")
    names = [m["name"] for m in resp2.json()["models"]]
    assert "del-m1" not in names


async def test_delete_model_not_found(client):
    """DELETE /v1/models/{name} returns 404 for unknown model."""
    resp = await client.delete("/v1/models/nonexistent-xyz")
    assert resp.status_code == 404


# ── 5. Model probe ──────────────────────────────────────────────────────────

async def test_probe_model(client):
    """POST /v1/models/{name}/probe returns specialty_scores."""
    _seed_test_model("probe-m1")
    resp = await client.post("/v1/models/probe-m1/probe")
    assert resp.status_code == 200
    data = resp.json()
    assert "specialty_scores" in data
    assert "model_name" in data
    assert data["model_name"] == "probe-m1"


async def test_probe_model_not_found(client):
    """POST /v1/models/{name}/probe returns 404 for unknown model."""
    resp = await client.post("/v1/models/nonexistent-xyz/probe")
    assert resp.status_code == 404


# ── 6. Stats simulate ───────────────────────────────────────────────────────

async def test_stats_simulate(client):
    """POST /v1/stats/simulate injects latency spike and may disable model."""
    _seed_test_model("sim-m1")
    resp = await client.post("/v1/stats/simulate", json={
        "model_name": "sim-m1",
        "latency_spike_ms": 5000.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    # Fields the dashboard JS reads
    assert "new_ema_latency" in data
    assert "disabled" in data
    assert "new_error_rate" in data
    assert "model" in data
    assert data["model"] == "sim-m1"
    # 10 injected data points at 5000ms vs 200ms expected * 5 = 1000ms
    # Should be disabled since EMA > 5x expected
    assert data["disabled"] is True


# ── 7. Config ───────────────────────────────────────────────────────────────

async def test_get_config(client):
    """GET /v1/config returns all YAML configs as top-level keys."""
    resp = await client.get("/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    # Should contain at least models and taxonomy from config dir
    assert isinstance(data, dict)
    # Config files exist, so at least some keys should be present
    assert len(data) > 0


async def test_config_reload(client):
    """POST /v1/config/reload reloads configs and returns status."""
    resp = await client.post("/v1/config/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reloaded"
    assert "loaded" in data


# ── 8. Sessions ─────────────────────────────────────────────────────────────

async def test_sessions_list(client):
    """GET /v1/sessions returns sessions list."""
    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


async def test_sessions_with_traces(client):
    """After an eval, sessions endpoint shows the session with traces."""
    _seed_test_model("sess-m1")
    # Create a trace via eval
    await client.post("/v1/eval", json={
        "query": "What is pathology?",
        "has_image": False,
        "user_id": "sess-test-user",
    })
    resp = await client.get("/v1/sessions")
    data = resp.json()
    assert len(data["sessions"]) >= 1
    # Each session should have expected fields
    s = data["sessions"][0]
    assert "session_id" in s
    assert "trace_count" in s


# ── 9. Prompts matrix ───────────────────────────────────────────────────────

async def test_prompt_matrix(client):
    """GET /v1/prompts/matrix returns matrix with model-specific overrides."""
    resp = await client.get("/v1/prompts/matrix")
    assert resp.status_code == 200
    data = resp.json()
    assert "matrix" in data
    assert isinstance(data["matrix"], list)
    # Each entry should have the fields the dashboard renders
    if data["matrix"]:
        entry = data["matrix"][0]
        assert "specialty" in entry
        assert "model_type" in entry
        assert "source" in entry
        assert "prompt_preview" in entry


# ── 10. Taxonomy / Specialties ──────────────────────────────────────────────

async def test_specialties(client):
    """GET /v1/specialties returns taxonomy tree and specialties list."""
    resp = await client.get("/v1/specialties")
    assert resp.status_code == 200
    data = resp.json()
    assert "taxonomy" in data
    assert "specialties" in data
    assert isinstance(data["specialties"], list)


# ── 11. Tools ───────────────────────────────────────────────────────────────

async def test_tools_list(client):
    """GET /v1/tools returns tool list with name and description."""
    resp = await client.get("/v1/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)
    if data["tools"]:
        tool = data["tools"][0]
        assert "name" in tool
        assert "description" in tool


# ── 12. Chat completions ────────────────────────────────────────────────────

async def test_chat_completions(client):
    """POST /v1/chat/completions with session_id and budget_strategy."""
    _seed_test_model("chat-m1")
    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What are the signs of heart failure?"}],
        "session_id": "test-session-001",
        "user": "dashboard-chat",
        "budget_strategy": "balanced",
    })
    assert resp.status_code == 200
    data = resp.json()
    # OpenAI-compatible response shape
    assert "choices" in data
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "content" in data["choices"][0]["message"]
    # Routing metadata
    assert "routing" in data
    routing = data["routing"]
    assert "model_selected" in routing
    assert "specialty_matched" in routing
    assert "total_latency_ms" in routing
    # Usage info
    assert "usage" in data
    assert "model" in data


async def test_chat_completions_no_budget(client):
    """POST /v1/chat/completions without budget_strategy defaults to auto."""
    _seed_test_model("chat-m2")
    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Explain MRI artifacts"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert "routing" in data


# ── 13. Traces ──────────────────────────────────────────────────────────────

async def test_traces_list(client):
    """GET /v1/traces returns trace list after eval queries."""
    _seed_test_model("trace-m1")
    # Generate a trace
    await client.post("/v1/eval", json={
        "query": "What is a drug interaction?",
        "has_image": False,
    })
    resp = await client.get("/v1/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert "traces" in data
    assert "count" in data
    assert data["count"] >= 1
    # Most recent trace should be first (reversed)
    t = data["traces"][0]
    assert "trace_id" in t
    assert "model_selected" in t
    assert "specialty_matched" in t


async def test_traces_limit(client):
    """GET /v1/traces?limit=1 limits number of returned traces."""
    _seed_test_model("trace-lim-m1")
    await client.post("/v1/eval", json={"query": "Q1", "has_image": False})
    await client.post("/v1/eval", json={"query": "Q2", "has_image": False})
    resp = await client.get("/v1/traces", params={"limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["traces"]) <= 1


# ── 14. Ambiguity detection ────────────────────────────────────────────────

async def test_eval_ambiguous_query(client):
    """POST /v1/eval with ambiguous query ("what is this?") returns low similarity."""
    _seed_test_model("ambig-m1")
    resp = await client.post("/v1/eval", json={
        "query": "what is this?",
        "has_image": False,
        "user_id": "ambig-user",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Simulated signals should return relatively low similarity
    assert "specialty_similarity" in data
    # The query is vague, so specialty sim should not be high
    assert data["specialty_similarity"] <= 0.8


# ── 15. Model enable/disable ───────────────────────────────────────────────

async def test_enable_disable_model(client):
    """POST /v1/models/{name}/enable and /disable toggle model state."""
    _seed_test_model("toggle-m1", enabled=True)

    # Disable
    resp = await client.post("/v1/models/toggle-m1/disable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "disabled"

    # Enable
    resp = await client.post("/v1/models/toggle-m1/enable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "enabled"


# ── 16. Stats models endpoint ──────────────────────────────────────────────

async def test_stats_models(client):
    """GET /v1/stats/models returns stats list and count."""
    resp = await client.get("/v1/stats/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "stats" in data
    assert "count" in data
    assert isinstance(data["stats"], list)


# ── 17. Session traces ─────────────────────────────────────────────────────

async def test_session_traces(client):
    """GET /v1/sessions/{id}/traces returns traces for a given session."""
    _seed_test_model("sesstr-m1")
    # Create a trace with a known session
    eval_resp = await client.post("/v1/eval", json={
        "query": "Test query for session trace",
        "has_image": False,
        "user_id": "sesstr-user",
    })
    trace = eval_resp.json()
    sid = trace["session_id"]

    resp = await client.get(f"/v1/sessions/{sid}/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert len(data["traces"]) >= 1


async def test_session_traces_not_found(client):
    """GET /v1/sessions/{id}/traces returns 404 for unknown session."""
    resp = await client.get("/v1/sessions/nonexistent-session/traces")
    assert resp.status_code == 404


# ── 18. Dashboard HTML served ───────────────────────────────────────────────

async def test_dashboard_served(client):
    """GET /dashboard serves the HTML dashboard."""
    resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert "MedVisionRouter" in resp.text


# ── 19. Specialty add (hot-add) ─────────────────────────────────────────────

async def test_add_specialty(client):
    """POST /v1/specialties adds a new specialty dynamically."""
    resp = await client.post("/v1/specialties", json={
        "name": "test_neurology",
        "title": "Test Neurology",
        "task_instruction": "Focus on neurological conditions.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "added"
    assert data["specialty"] == "test_neurology"


# ── 20. Eval query flow mirrors dashboard sendQuery() ───────────────────────

async def test_eval_matches_dashboard_sendquery(client):
    """Verify the exact request shape sent by the dashboard's sendQuery() function."""
    _seed_test_model("dash-m1")
    # Exact shape from dashboard JS: {query, has_image, image_base64, user_id}
    resp = await client.post("/v1/eval", json={
        "query": "What are the current ACC/AHA guidelines for heart failure?",
        "has_image": False,
        "image_base64": None,
        "user_id": "dashboard-user",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Dashboard renders these fields in addTrace()
    assert "trace_id" in data
    assert "query_preview" in data
    assert "specialty_matched" in data
    assert "model_selected" in data
    assert "model_type" in data
    assert "safety_score" in data
    assert "complexity_score" in data
    assert "image_type_detected" in data
    assert "tools_suggested" in data
    assert "specialty_similarity" in data
    assert "cost_estimate" in data
    assert "prompt_source" in data
    assert "reasoning_tokens" in data
    assert "prompt_used" in data
    assert "signal_latency_ms" in data
    assert "routing_latency_ms" in data
    assert "total_latency_ms" in data


# ── 21. Chat completions mirrors dashboard chatSend() ───────────────────────

async def test_chat_matches_dashboard_chatsend(client):
    """Verify the exact request/response shape used by the dashboard's chatSend()."""
    _seed_test_model("dchat-m1")
    # Exact shape from dashboard JS
    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is a cardiac murmur?"}],
        "session_id": "chat-test-session",
        "user": "dashboard-chat",
        "budget_strategy": "quality_first",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Dashboard reads: data.choices[0].message.content, data.routing
    assert data["choices"][0]["message"]["content"]
    routing = data["routing"]
    # Dashboard reads these from routing:
    assert "specialty_matched" in routing
    assert "model_selected" in routing
    assert "total_latency_ms" in routing
    assert "cost_estimate" in routing
    assert "complexity_score" in routing
    assert "safety_score" in routing
