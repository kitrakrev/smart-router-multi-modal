"""Shared fixtures for MedVisionRouter API tests."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.server import app, _trace_log
from src.registry.models import model_registry, ModelEntry
from src.registry.stats import stats_tracker
from src.memory.session import session_store


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global state between tests so they are isolated."""
    # Snapshot current state
    old_models = dict(model_registry._models)
    old_stats = dict(stats_tracker._stats)
    old_expected = dict(stats_tracker._expected_latency)
    old_sessions = dict(session_store._sessions)
    old_traces = list(_trace_log)

    yield

    # Restore
    model_registry._models = old_models
    stats_tracker._stats = old_stats
    stats_tracker._expected_latency = old_expected
    session_store._sessions = old_sessions
    _trace_log.clear()
    _trace_log.extend(old_traces)


def _seed_test_model(
    name: str = "test-model",
    model_type: str = "generalist",
    capabilities: list[str] | None = None,
    quality: float = 0.8,
    enabled: bool = True,
) -> ModelEntry:
    """Helper to add a test model to the registry."""
    entry = ModelEntry(
        name=name,
        type=model_type,
        provider="local",
        model_id=name,
        capabilities=capabilities or ["text"],
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        avg_latency_ms=200.0,
        quality_score=quality,
        enabled=enabled,
    )
    model_registry.add_model(entry)
    stats_tracker.set_expected_latency(name, entry.avg_latency_ms)
    return entry


@pytest_asyncio.fixture
async def client():
    """Provide an async httpx client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
