"""FastAPI router for Model Registry CRUD + health + discovery endpoints.

Mount this as a sub-router in server.py:
    from models_api import models_router
    app.include_router(models_router)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models import model_registry, ModelEntry, ModelCapability, _parse_capabilities

models_router = APIRouter(prefix="/v1/models", tags=["models"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class AddModelRequest(BaseModel):
    name: str
    provider: str = "custom"
    api_base: str = ""
    api_key: str = ""
    model_id: str = ""
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 500.0
    max_context: int = 128000
    enabled: bool = True


class UpdateModelRequest(BaseModel):
    provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    capabilities: Optional[list[str]] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    max_context: Optional[int] = None
    enabled: Optional[bool] = None


class DiscoverRequest(BaseModel):
    api_base: str
    api_key: str = ""
    provider: str = "custom"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@models_router.get("")
async def list_models():
    """GET /v1/models -- list all registered models."""
    entries = await model_registry.list_models()
    return {
        "object": "list",
        "data": [e.to_dict() for e in entries],
    }


@models_router.post("")
async def add_model(req: AddModelRequest):
    """POST /v1/models -- register a new model."""
    caps = _parse_capabilities(req.capabilities)
    entry = ModelEntry(
        name=req.name,
        provider=req.provider,
        api_base=req.api_base,
        api_key=req.api_key,
        model_id=req.model_id or req.name,
        capabilities=caps,
        cost_per_1k_input=req.cost_per_1k_input,
        cost_per_1k_output=req.cost_per_1k_output,
        avg_latency_ms=req.avg_latency_ms,
        max_context=req.max_context,
        enabled=req.enabled,
    )
    ok = await model_registry.add_model(entry)
    if not ok:
        raise HTTPException(status_code=409, detail=f"Model '{req.name}' already exists")
    return {"status": "added", "model": entry.to_dict()}


@models_router.put("/{name}")
async def update_model(name: str, req: UpdateModelRequest):
    """PUT /v1/models/{name} -- update an existing model."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = await model_registry.update_model(name, updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    entry = await model_registry.get_model(name)
    return {"status": "updated", "model": entry.to_dict() if entry else {}}


@models_router.delete("/{name}")
async def remove_model(name: str):
    """DELETE /v1/models/{name} -- remove a model."""
    ok = await model_registry.remove_model(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"status": "removed", "name": name}


@models_router.get("/{name}/health")
async def check_health(name: str):
    """GET /v1/models/{name}/health -- check model endpoint reachability."""
    result = await model_registry.check_health(name)
    if result.get("error") == "model not found":
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return result


@models_router.post("/discover")
async def discover_models(req: DiscoverRequest):
    """POST /v1/models/discover -- auto-discover from OpenAI-compatible endpoint."""
    discovered = await model_registry.discover_models(
        api_base=req.api_base,
        api_key=req.api_key,
        provider=req.provider,
    )
    return {
        "status": "discovered",
        "count": len(discovered),
        "models": [e.to_dict() for e in discovered],
    }
