"""Dynamic Model Registry for the LLM Router.

Provides runtime add/remove/update of LLM model entries with capability
filtering, cost-based selection, and health checking.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx


# ---------------------------------------------------------------------------
# Enums & Data models
# ---------------------------------------------------------------------------

class ModelCapability(str, Enum):
    TEXT = "text"
    VISION = "vision"
    TOOLS = "tools"
    REASONING = "reasoning"
    CODE = "code"


@dataclass
class ModelEntry:
    name: str                           # "gpt-4o"
    provider: str                       # "openai" | "anthropic" | "local" | "custom"
    api_base: str                       # "https://api.openai.com/v1"
    api_key: str                        # env var name or direct key
    model_id: str                       # actual model ID to send in API call
    capabilities: set[ModelCapability]  # {TEXT, VISION, TOOLS}
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 500.0
    max_context: int = 128000
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": _mask_key(self.api_key),
            "model_id": self.model_id,
            "capabilities": sorted(c.value for c in self.capabilities),
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "avg_latency_ms": self.avg_latency_ms,
            "max_context": self.max_context,
            "enabled": self.enabled,
        }


@dataclass
class ModelRuntimeStats:
    """Runtime performance statistics for a single model."""
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    latency_ema: float = 0.0
    error_rate_ema: float = 0.0
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_by_domain: dict = field(default_factory=dict)
    last_health_check: datetime = field(default_factory=datetime.now)
    status: str = "healthy"  # healthy | degraded | disabled | unknown
    disabled_reason: str = ""
    disabled_at: Optional[datetime] = None

    # Thresholds (configurable per model)
    latency_threshold_ms: float = 5000
    error_threshold: float = 0.2
    recovery_check_interval_s: int = 30


def _mask_key(key: str) -> str:
    """Mask API key for display, showing only last 4 chars."""
    if not key or len(key) <= 8:
        return "****"
    return f"****{key[-4:]}"


def _parse_capabilities(raw: list[str] | set[str] | None) -> set[ModelCapability]:
    """Parse capability strings into ModelCapability enum values."""
    if not raw:
        return {ModelCapability.TEXT}
    result = set()
    for c in raw:
        try:
            result.add(ModelCapability(c.lower().strip()))
        except ValueError:
            pass  # skip unknown capabilities
    return result if result else {ModelCapability.TEXT}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe in-memory registry of LLM model entries."""

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}
        self._lock = asyncio.Lock()
        self._health_cache: dict[str, dict] = {}  # name -> {healthy, checked_at, latency_ms}
        self.runtime_stats: dict[str, ModelRuntimeStats] = {}

    # -- CRUD ---------------------------------------------------------------

    async def add_model(self, entry: ModelEntry) -> bool:
        """Add a new model. Returns False if name already exists."""
        async with self._lock:
            if entry.name in self._models:
                return False
            self._models[entry.name] = entry
            self.runtime_stats[entry.name] = ModelRuntimeStats()
            return True

    async def remove_model(self, name: str) -> bool:
        """Remove a model by name. Returns False if not found."""
        async with self._lock:
            if name not in self._models:
                return False
            del self._models[name]
            self._health_cache.pop(name, None)
            self.runtime_stats.pop(name, None)
            return True

    async def update_model(self, name: str, updates: dict) -> bool:
        """Update fields on an existing model. Returns False if not found."""
        async with self._lock:
            entry = self._models.get(name)
            if not entry:
                return False
            for k, v in updates.items():
                if k == "capabilities":
                    entry.capabilities = _parse_capabilities(v)
                elif k == "enabled":
                    entry.enabled = bool(v)
                elif hasattr(entry, k) and k != "name":
                    setattr(entry, k, v)
            return True

    async def list_models(self) -> list[ModelEntry]:
        """Return all registered models."""
        async with self._lock:
            return list(self._models.values())

    async def get_model(self, name: str) -> Optional[ModelEntry]:
        async with self._lock:
            return self._models.get(name)

    async def get_capable_models(
        self, required: set[ModelCapability], enabled_only: bool = True
    ) -> list[ModelEntry]:
        """Return models that have ALL of the required capabilities."""
        async with self._lock:
            results = []
            for m in self._models.values():
                if enabled_only and not m.enabled:
                    continue
                if required.issubset(m.capabilities):
                    results.append(m)
            return results

    async def get_cheapest_capable(
        self, required: set[ModelCapability], enabled_only: bool = True
    ) -> Optional[ModelEntry]:
        """Return the cheapest model that has all required capabilities."""
        capable = await self.get_capable_models(required, enabled_only)
        if not capable:
            return None
        return min(capable, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

    # -- Health check -------------------------------------------------------

    async def check_health(self, name: str) -> dict:
        """Ping a model's API endpoint to check reachability.

        Returns dict with: healthy (bool), latency_ms, error, checked_at
        """
        entry = await self.get_model(name)
        if not entry:
            return {"healthy": False, "error": "model not found", "latency_ms": 0, "checked_at": time.time()}

        result = {"healthy": False, "latency_ms": 0, "checked_at": time.time(), "error": None}
        try:
            url = entry.api_base.rstrip("/")
            # Try /models endpoint (OpenAI-compatible)
            if not url.endswith("/models"):
                url = url + "/models"
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {}
                if entry.api_key and not entry.api_key.startswith("$"):
                    headers["Authorization"] = f"Bearer {entry.api_key}"
                resp = await client.get(url, headers=headers)
            latency = round((time.perf_counter() - t0) * 1000, 1)
            result["latency_ms"] = latency
            result["healthy"] = resp.status_code in (200, 401, 403)  # auth errors still mean reachable
            if resp.status_code >= 500:
                result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)

        self._health_cache[name] = result
        return result

    def get_cached_health(self, name: str) -> Optional[dict]:
        return self._health_cache.get(name)

    # -- Runtime stats & adaptive routing ------------------------------------

    def update_stats(self, model_name: str, latency_ms: float, success: bool, domain: str | None = None) -> ModelRuntimeStats | None:
        """Called after EVERY model response. Updates EMA stats."""
        stats = self.runtime_stats.get(model_name)
        if stats is None:
            return None
        alpha = 0.1  # EMA smoothing factor

        stats.total_requests += 1
        if success:
            stats.total_successes += 1
        else:
            stats.total_failures += 1

        stats.latency_ema = (1 - alpha) * stats.latency_ema + alpha * latency_ms
        stats.error_rate_ema = (1 - alpha) * stats.error_rate_ema + alpha * (0 if success else 1)
        stats.latency_history.append(latency_ms)

        if domain:
            domain_stats = stats.accuracy_by_domain.setdefault(domain, {'correct': 0, 'total': 0})
            domain_stats['total'] += 1
            if success:
                domain_stats['correct'] += 1

        # Auto-disable check
        model = self._models.get(model_name)
        if model:
            if stats.latency_ema > model.avg_latency_ms * 3:
                stats.status = "degraded"
                if stats.latency_ema > model.avg_latency_ms * 5:
                    model.enabled = False
                    stats.status = "disabled"
                    stats.disabled_reason = f"Latency EMA {stats.latency_ema:.0f}ms exceeds 5x threshold"
                    stats.disabled_at = datetime.now()
            elif stats.error_rate_ema <= stats.error_threshold and stats.status == "degraded":
                stats.status = "healthy"

            if stats.error_rate_ema > stats.error_threshold:
                model.enabled = False
                stats.status = "disabled"
                stats.disabled_reason = f"Error rate {stats.error_rate_ema:.1%} exceeds threshold"
                stats.disabled_at = datetime.now()

        return stats

    async def recovery_check_loop(self):
        """Background task: periodically check if disabled models recovered."""
        while True:
            await asyncio.sleep(30)
            for name, model in list(self._models.items()):
                if not model.enabled and self.runtime_stats.get(name, ModelRuntimeStats()).status == "disabled":
                    try:
                        result = await self.check_health(name)
                        healthy = result.get("healthy", False)
                    except Exception:
                        healthy = False
                    if healthy:
                        model.enabled = True
                        stats = self.runtime_stats.get(name)
                        if stats:
                            stats.status = "healthy"
                            stats.latency_ema *= 0.5
                            stats.error_rate_ema *= 0.5
                            stats.disabled_reason = ""
                            stats.disabled_at = None

    def get_stats_snapshot(self) -> dict:
        """Return all model stats for dashboard."""
        result = {}
        for name, stats in self.runtime_stats.items():
            sorted_history = sorted(stats.latency_history) if stats.latency_history else []
            hist_len = len(sorted_history)
            result[name] = {
                'status': stats.status,
                'total_requests': stats.total_requests,
                'total_successes': stats.total_successes,
                'total_failures': stats.total_failures,
                'latency_ema': round(stats.latency_ema, 1),
                'error_rate': round(stats.error_rate_ema, 3),
                'latency_p50': sorted_history[hist_len // 2] if hist_len > 0 else 0,
                'latency_p95': sorted_history[int(0.95 * hist_len)] if hist_len > 0 else 0,
                'latency_history': list(stats.latency_history),
                'accuracy_by_domain': {
                    d: round(s['correct'] / s['total'], 3) if s['total'] > 0 else 0
                    for d, s in stats.accuracy_by_domain.items()
                },
                'disabled_reason': stats.disabled_reason,
            }
        return result

    # -- Auto-discover from OpenAI-compatible endpoints ---------------------

    async def discover_models(
        self, api_base: str, api_key: str = "", provider: str = "custom"
    ) -> list[ModelEntry]:
        """Query an OpenAI-compatible /models endpoint and register discovered models."""
        url = api_base.rstrip("/") + "/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        discovered = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            models_list = data.get("data", []) if isinstance(data, dict) else data
            for m in models_list:
                model_id = m.get("id", "")
                if not model_id:
                    continue
                name = model_id.replace("/", "-").replace(":", "-")
                entry = ModelEntry(
                    name=name,
                    provider=provider,
                    api_base=api_base,
                    api_key=api_key,
                    model_id=model_id,
                    capabilities={ModelCapability.TEXT},
                    cost_per_1k_input=0.0,
                    cost_per_1k_output=0.0,
                    avg_latency_ms=500.0,
                    max_context=m.get("context_length", 128000),
                    enabled=True,
                )
                added = await self.add_model(entry)
                if added:
                    discovered.append(entry)
        except Exception:
            pass  # caller should handle empty results
        return discovered

    # -- Seed from config.yaml models list ----------------------------------

    async def seed_from_config(self, models_config: list[dict]):
        """Populate registry from the config.yaml models list."""
        for m in models_config:
            caps = _parse_capabilities(m.get("capabilities", []))
            # Map old generic caps to new enum
            raw_caps = set(m.get("capabilities", []))
            if "general" in raw_caps or "fast" in raw_caps:
                caps.add(ModelCapability.TEXT)
            entry = ModelEntry(
                name=m["name"],
                provider=m.get("provider", "custom"),
                api_base=m.get("api_base", ""),
                api_key=m.get("api_key", ""),
                model_id=m.get("model_id", m["name"]),
                capabilities=caps,
                cost_per_1k_input=m.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=m.get("cost_per_1k_output", 0.0),
                avg_latency_ms=m.get("avg_latency_ms", 500.0),
                max_context=m.get("max_context", 128000),
                enabled=m.get("enabled", True),
            )
            await self.add_model(entry)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

model_registry = ModelRegistry()
