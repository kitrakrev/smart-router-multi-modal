"""Session tracking with trace timeline."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionTrace:
    """A single trace within a session."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    query_preview: str = ""
    has_image: bool = False

    # Signal results
    specialty_matched: str = ""
    specialty_similarity: float = 0.0
    image_type_detected: str = ""
    complexity_score: float = 0.0
    safety_score: float = 0.0
    tools_suggested: list[str] = field(default_factory=list)

    # Routing decision
    model_selected: str = ""
    model_type: str = ""
    prompt_used: str = ""
    reasoning_tokens: int = 0
    cost_estimate: float = 0.0

    # Latency breakdown
    signal_latency_ms: dict[str, float] = field(default_factory=dict)
    routing_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Result
    success: bool = True
    response_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query_preview": self.query_preview,
            "has_image": self.has_image,
            "specialty_matched": self.specialty_matched,
            "specialty_similarity": round(self.specialty_similarity, 3),
            "image_type_detected": self.image_type_detected,
            "complexity_score": round(self.complexity_score, 2),
            "safety_score": round(self.safety_score, 2),
            "tools_suggested": self.tools_suggested,
            "model_selected": self.model_selected,
            "model_type": self.model_type,
            "prompt_used": self.prompt_used,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_estimate": round(self.cost_estimate, 6),
            "signal_latency_ms": {
                k: round(v, 1) for k, v in self.signal_latency_ms.items()
            },
            "routing_latency_ms": round(self.routing_latency_ms, 1),
            "model_latency_ms": round(self.model_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "success": self.success,
            "response_preview": self.response_preview,
        }


@dataclass
class Session:
    """A user session containing multiple traces."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    traces: list[SessionTrace] = field(default_factory=list)

    def add_trace(self, trace: SessionTrace) -> None:
        self.traces.append(trace)
        self.last_active = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "trace_count": len(self.traces),
            "traces": [t.to_dict() for t in self.traces],
        }

    def summary(self) -> dict[str, Any]:
        """Short summary without full trace data."""
        specialties: dict[str, int] = {}
        for t in self.traces:
            s = t.specialty_matched or "unknown"
            specialties[s] = specialties.get(s, 0) + 1
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "trace_count": len(self.traces),
            "specialty_distribution": specialties,
        }


class SessionStore:
    """In-memory session store with size limit."""

    MAX_SESSIONS = 500

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get_or_create(self, session_id: str, user_id: str = "") -> Session:
        if session_id not in self._sessions:
            session = Session(session_id=session_id, user_id=user_id)
            self._sessions[session_id] = session
            # Evict oldest if over limit
            while len(self._sessions) > self.MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent session summaries."""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        return [s.summary() for s in sessions[:limit]]

    def get_traces(self, session_id: str) -> list[dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return [t.to_dict() for t in session.traces]

    def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return [
            s.summary()
            for s in self._sessions.values()
            if s.user_id == user_id
        ]


# Singleton
session_store = SessionStore()
