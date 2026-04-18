"""Model registry with dynamic add/remove/discover."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A registered model with its metadata."""

    name: str
    type: str  # specialist | generalist | reasoning
    provider: str  # local | openai | anthropic | google
    model_id: str
    capabilities: list[str] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 500.0
    quality_score: float = 0.7
    enabled: bool = True
    approved: bool = True  # human-approved for production use
    specialties: list[str] = field(default_factory=list)
    added_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "provider": self.provider,
            "model_id": self.model_id,
            "capabilities": self.capabilities,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "avg_latency_ms": self.avg_latency_ms,
            "quality_score": self.quality_score,
            "enabled": self.enabled,
            "approved": self.approved,
            "specialties": self.specialties,
            "added_at": self.added_at,
        }


class ModelRegistry:
    """Dynamic model registry supporting add/remove/discover."""

    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}

    # -- CRUD ------------------------------------------------------------------

    def add_model(self, entry: ModelEntry) -> None:
        logger.info("Registry: adding model %s", entry.name)
        self._models[entry.name] = entry

    def remove_model(self, name: str) -> bool:
        if name in self._models:
            del self._models[name]
            logger.info("Registry: removed model %s", name)
            return True
        return False

    def update_model(self, name: str, **kwargs: Any) -> bool:
        entry = self._models.get(name)
        if entry is None:
            return False
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return True

    def get_model(self, name: str) -> Optional[ModelEntry]:
        return self._models.get(name)

    def list_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    # -- Capability queries ----------------------------------------------------

    def get_capable_models(self, required_capabilities: list[str]) -> list[ModelEntry]:
        """Return models that have ALL required capabilities and are enabled."""
        result = []
        req = set(required_capabilities)
        for m in self._models.values():
            if not m.enabled or not m.approved:
                continue
            if req.issubset(set(m.capabilities)):
                result.append(m)
        return result

    def get_cheapest_capable(self, required: list[str]) -> Optional[ModelEntry]:
        """Return the cheapest model that has all required capabilities."""
        capable = self.get_capable_models(required)
        if not capable:
            return None
        return min(capable, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

    def get_models_for_specialty(self, specialty: str) -> list[ModelEntry]:
        """Return models that list this specialty (or have 'medical' capability for med specialties)."""
        result = []
        for m in self._models.values():
            if not m.enabled or not m.approved:
                continue
            if specialty in m.specialties or "medical" in m.capabilities:
                result.append(m)
        return result

    # -- Discovery -------------------------------------------------------------

    async def discover_from_endpoint(self, api_base: str) -> list[ModelEntry]:
        """Auto-discover models from an OpenAI-compatible /v1/models endpoint."""
        if httpx is None:
            logger.warning("httpx not installed, cannot discover models")
            return []

        discovered: list[ModelEntry] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{api_base.rstrip('/')}/v1/models"
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("data", []):
                model_id = item.get("id", "")
                name = model_id.split("/")[-1] if "/" in model_id else model_id
                if name in self._models:
                    continue
                entry = ModelEntry(
                    name=name,
                    type="generalist",
                    provider="discovered",
                    model_id=model_id,
                    capabilities=["text"],
                    approved=False,  # needs human approval
                )
                self.add_model(entry)
                discovered.append(entry)
                logger.info("Discovered model: %s", name)
        except Exception as exc:
            logger.error("Model discovery failed for %s: %s", api_base, exc)

        return discovered

    # -- Config loading --------------------------------------------------------

    def seed_from_config(self, config_path: str | Path) -> int:
        """Load models from a models.yaml config file. Returns count loaded."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Models config not found: %s", path)
            return 0

        with open(path) as f:
            cfg = yaml.safe_load(f)

        count = 0
        for m in cfg.get("models", []):
            entry = ModelEntry(
                name=m["name"],
                type=m.get("type", "generalist"),
                provider=m.get("provider", "unknown"),
                model_id=m.get("model_id", m["name"]),
                capabilities=m.get("capabilities", ["text"]),
                cost_per_1k_input=m.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=m.get("cost_per_1k_output", 0.0),
                avg_latency_ms=m.get("avg_latency_ms", 500.0),
                quality_score=m.get("quality_score", 0.7),
            )
            self.add_model(entry)
            count += 1

        logger.info("Seeded %d models from %s", count, path)
        return count


# Singleton
model_registry = ModelRegistry()
