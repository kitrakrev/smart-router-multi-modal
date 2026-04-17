"""Request tracing and WebSocket live streaming.

Maintains an in-memory store of the last 1000 traces and broadcasts
new traces to all connected WebSocket subscribers.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque, OrderedDict
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
    session_id: str = ""
    user_id: str = ""

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
            "session_id": self.session_id,
            "user_id": self.user_id,
        }


# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    created_at: str
    last_active: float  # time.time() for expiry checks
    trace_ids: list[str] = field(default_factory=list)
    model_stats_snapshot_start: dict = field(default_factory=dict)

    def to_dict(self, traces: list[RequestTrace]) -> dict:
        session_traces = [t for t in traces if t.trace_id in set(self.trace_ids)]
        total_cost = sum(t.decision.get("estimated_cost", 0) for t in session_traces if not t.decision.get("blocked"))
        model_counts: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        blocked_count = 0
        pii_count = 0
        total_routing_ms = 0.0
        total_model_ms = 0.0
        model_call_count = 0
        for t in session_traces:
            d = t.decision
            if d.get("blocked"):
                blocked_count += 1
                model_counts["BLOCKED"] = model_counts.get("BLOCKED", 0) + 1
            else:
                m = d.get("selected_model", "unknown")
                model_counts[m] = model_counts.get(m, 0) + 1
                dn = d.get("decision_name", "")
                if dn:
                    decision_counts[dn] = decision_counts.get(dn, 0) + 1
                if t.model_response_latency_ms > 0:
                    total_model_ms += t.model_response_latency_ms
                    model_call_count += 1
            total_routing_ms += t.total_latency_ms
            for s in t.signals:
                if s.get("name") == "pii" and s.get("metadata", {}).get("total_matches", 0) > 0:
                    pii_count += 1

        avg_routing = total_routing_ms / len(session_traces) if session_traces else 0
        avg_model = total_model_ms / model_call_count if model_call_count > 0 else 0

        # Cost savings estimate: compare to all-claude
        premium_cost_per_1k = 0.01
        cost_without = 0
        for t in session_traces:
            if not t.decision.get("blocked"):
                tokens = len((t.query or "").split()) * 1.3
                cost_without += (tokens / 1000) * premium_cost_per_1k
        savings_pct = ((cost_without - total_cost) / cost_without * 100) if cost_without > 0 else 0

        idle_sec = time.time() - self.last_active
        status = "active" if idle_sec < 300 else "idle"

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "query_count": len(session_traces),
            "total_cost": round(total_cost, 6),
            "cost_without_router": round(cost_without, 6),
            "savings_pct": round(savings_pct, 1),
            "model_counts": model_counts,
            "decision_counts": decision_counts,
            "blocked_count": blocked_count,
            "pii_count": pii_count,
            "avg_routing_ms": round(avg_routing, 1),
            "avg_model_ms": round(avg_model, 1),
            "status": status,
            "last_active": self.last_active,
        }


class SessionStore:
    """In-memory session store with expiry and max capacity."""

    def __init__(self, max_sessions: int = 100, expiry_seconds: int = 1800):
        self.sessions: OrderedDict[str, SessionRecord] = OrderedDict()
        self.max_sessions = max_sessions
        self.expiry_seconds = expiry_seconds

    def get_or_create(self, session_id: str, user_id: str, model_stats_snapshot: dict | None = None) -> SessionRecord:
        now = time.time()
        self._evict_expired(now)

        if session_id in self.sessions:
            s = self.sessions[session_id]
            s.last_active = now
            if user_id and not s.user_id:
                s.user_id = user_id
            # Move to end (most recent)
            self.sessions.move_to_end(session_id)
            return s

        # Create new session
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest
            self.sessions.popitem(last=False)

        s = SessionRecord(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_active=now,
            model_stats_snapshot_start=model_stats_snapshot or {},
        )
        self.sessions[session_id] = s
        return s

    def add_trace(self, session_id: str, trace_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].trace_ids.append(trace_id)
            self.sessions[session_id].last_active = time.time()

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def get_all(self) -> list[SessionRecord]:
        self._evict_expired(time.time())
        return list(reversed(self.sessions.values()))

    def get_sessions_for_user(self, user_id: str) -> list[SessionRecord]:
        return [s for s in self.sessions.values() if s.user_id == user_id]

    def _evict_expired(self, now: float):
        expired = [sid for sid, s in self.sessions.items() if now - s.last_active > self.expiry_seconds]
        for sid in expired:
            del self.sessions[sid]


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
        self.sessions = SessionStore()

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
        session_id: str = "",
        user_id: str = "",
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

        # Auto-generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        trace = RequestTrace(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query[:100],
            signals=sig_dicts,
            decision=dec_dict,
            total_latency_ms=round(total_latency_ms, 2),
            model_response_latency_ms=round(model_response_latency_ms, 2),
            request_number=self._counter,
            session_id=session_id,
            user_id=user_id,
        )

        # Track session
        model_stats_snap = self._registry.get_stats_snapshot() if self._registry else {}
        self.sessions.get_or_create(session_id, user_id, model_stats_snapshot=model_stats_snap)
        self.sessions.add_trace(session_id, trace_id)

        async with self._lock:
            self.traces.append(trace)

        # Broadcast to WebSocket subscribers
        await self._broadcast(trace)
        return trace

    async def _broadcast(self, trace: RequestTrace):
        message = {"type": "new_trace", "trace": trace.to_dict()}
        if self._registry and hasattr(self._registry, 'get_stats_snapshot'):
            message["model_stats"] = self._registry.get_stats_snapshot()
        message["sessions"] = self.get_all_sessions()
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

    def get_traces_for_session(self, session_id: str) -> list[dict]:
        return [t.to_dict() for t in self.traces if t.session_id == session_id]

    def get_traces_for_user(self, user_id: str) -> list[dict]:
        return [t.to_dict() for t in self.traces if t.user_id == user_id]

    def get_user_stats(self, user_id: str) -> dict:
        user_traces = [t for t in self.traces if t.user_id == user_id]
        if not user_traces:
            return {"user_id": user_id, "total_requests": 0}
        total = len(user_traces)
        model_dist: dict[str, int] = {}
        decision_dist: dict[str, int] = {}
        blocked = 0
        total_cost = 0.0
        total_lat = 0.0
        for t in user_traces:
            d = t.decision
            if d.get("blocked"):
                blocked += 1
            else:
                m = d.get("selected_model", "unknown")
                model_dist[m] = model_dist.get(m, 0) + 1
                dn = d.get("decision_name", "")
                if dn:
                    decision_dist[dn] = decision_dist.get(dn, 0) + 1
                total_cost += d.get("estimated_cost", 0)
            total_lat += t.total_latency_ms
        sessions = self.sessions.get_sessions_for_user(user_id)
        return {
            "user_id": user_id,
            "total_requests": total,
            "total_sessions": len(sessions),
            "blocked_count": blocked,
            "total_cost": round(total_cost, 6),
            "avg_latency_ms": round(total_lat / total, 2),
            "model_distribution": model_dist,
            "decision_distribution": decision_dist,
        }

    def get_all_sessions(self) -> list[dict]:
        all_traces = list(self.traces)
        sessions = self.sessions.get_all()
        return [s.to_dict(all_traces) for s in sessions]

    def get_session_adaptive_updates(self, session_id: str) -> list[dict]:
        """Compare model stats at session start vs current stats."""
        session = self.sessions.get_session(session_id)
        if not session or not self._registry:
            return []
        start_snap = session.model_stats_snapshot_start
        current_snap = self._registry.get_stats_snapshot()
        updates = []
        for model_name, current in current_snap.items():
            start = start_snap.get(model_name, {})
            # Latency EMA change
            start_ema = start.get("latency_ema", 0)
            cur_ema = current.get("latency_ema", 0)
            if start_ema > 0 and abs(cur_ema - start_ema) > 20:
                updates.append({
                    "type": "latency_change",
                    "model": model_name,
                    "message": f"{model_name} latency EMA: {start_ema:.0f}ms -> {cur_ema:.0f}ms",
                    "old_value": round(start_ema, 1),
                    "new_value": round(cur_ema, 1),
                })
            # Error rate change
            start_err = start.get("error_rate", 0)
            cur_err = current.get("error_rate", 0)
            if abs(cur_err - start_err) > 0.01:
                updates.append({
                    "type": "error_rate_change",
                    "model": model_name,
                    "message": f"{model_name} error_rate: {start_err:.1%} -> {cur_err:.1%}",
                    "old_value": round(start_err, 4),
                    "new_value": round(cur_err, 4),
                })
            # Status change
            start_status = start.get("status", "unknown")
            cur_status = current.get("status", "unknown")
            if start_status != cur_status and start_status != "unknown":
                updates.append({
                    "type": "status_change",
                    "model": model_name,
                    "message": f"{model_name} status: {start_status} -> {cur_status}",
                    "old_value": start_status,
                    "new_value": cur_status,
                })
        return updates

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
