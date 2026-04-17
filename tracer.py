"""Request tracing and WebSocket live streaming.

Maintains an in-memory store of the last 1000 traces and broadcasts
new traces to all connected WebSocket subscribers.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import WebSocket

from signals import SignalResult
from router import RouterDecision


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    trace_id: str
    timestamp: str
    query: str  # first 100 chars
    signals: list[dict]
    decision: dict
    total_latency_ms: float
    model_response_latency_ms: float
    request_number: int = 0

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "signals": self.signals,
            "decision": self.decision,
            "total_latency_ms": self.total_latency_ms,
            "model_response_latency_ms": self.model_response_latency_ms,
            "request_number": self.request_number,
        }


# ---------------------------------------------------------------------------
# Trace store
# ---------------------------------------------------------------------------

class TraceStore:
    def __init__(self, max_traces: int = 1000):
        self.traces: deque[RequestTrace] = deque(maxlen=max_traces)
        self.subscribers: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        self._registry = None  # set via set_registry()

    def set_registry(self, registry):
        """Attach a ModelRegistry for stats broadcasting."""
        self._registry = registry

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    async def add_trace(
        self,
        trace_id: str,
        query: str,
        signals: list[SignalResult],
        decision: RouterDecision,
        total_latency_ms: float,
        model_response_latency_ms: float = 0.0,
    ) -> RequestTrace:
        self._counter += 1

        sig_dicts = []
        for s in signals:
            sig_dicts.append({
                "name": s.name,
                "score": s.score,
                "confidence": s.confidence,
                "execution_time_ms": s.execution_time_ms,
                "metadata": s.metadata,
                "skipped": s.skipped,
            })

        dec_dict = {
            "selected_model": decision.selected_model,
            "inference_config": decision.inference_config,
            "estimated_cost": decision.estimated_cost,
            "estimated_latency_ms": decision.estimated_latency_ms,
            "reason": decision.reason,
            "blocked": decision.blocked,
            "decision_name": getattr(decision, "decision_name", ""),
            "similarity": getattr(decision, "similarity", 0.0),
        }

        trace = RequestTrace(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query[:100],
            signals=sig_dicts,
            decision=dec_dict,
            total_latency_ms=round(total_latency_ms, 2),
            model_response_latency_ms=round(model_response_latency_ms, 2),
            request_number=self._counter,
        )

        async with self._lock:
            self.traces.append(trace)

        # Broadcast to WebSocket subscribers
        await self._broadcast(trace)
        return trace

    async def _broadcast(self, trace: RequestTrace):
        message = {"type": "new_trace", "trace": trace.to_dict()}
        if self._registry and hasattr(self._registry, 'get_stats_snapshot'):
            message["model_stats"] = self._registry.get_stats_snapshot()
        payload = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self.subscribers:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self.subscribers.remove(ws)
            except ValueError:
                pass

    async def subscribe(self, ws: WebSocket):
        self.subscribers.append(ws)

    async def unsubscribe(self, ws: WebSocket):
        try:
            self.subscribers.remove(ws)
        except ValueError:
            pass

    def get_trace(self, trace_id: str) -> Optional[RequestTrace]:
        for t in reversed(self.traces):
            if t.trace_id == trace_id:
                return t
        return None

    def get_recent(self, n: int = 50) -> list[dict]:
        items = list(self.traces)[-n:]
        return [t.to_dict() for t in reversed(items)]

    def get_stats(self) -> dict:
        if not self.traces:
            return {"total_requests": 0, "avg_latency_ms": 0, "model_distribution": {}, "avg_signal_times": {}}

        total = len(self.traces)
        avg_lat = sum(t.total_latency_ms for t in self.traces) / total

        model_dist: dict[str, int] = {}
        sig_times: dict[str, list[float]] = {}
        for t in self.traces:
            m = t.decision.get("selected_model", "unknown")
            model_dist[m] = model_dist.get(m, 0) + 1
            for s in t.signals:
                name = s.get("name", "unknown")
                if name not in sig_times:
                    sig_times[name] = []
                sig_times[name].append(s.get("execution_time_ms", 0))

        avg_sig = {k: round(sum(v) / len(v), 2) for k, v in sig_times.items()}

        return {
            "total_requests": total,
            "avg_latency_ms": round(avg_lat, 2),
            "model_distribution": model_dist,
            "avg_signal_times": avg_sig,
        }


# Singleton
trace_store = TraceStore()
