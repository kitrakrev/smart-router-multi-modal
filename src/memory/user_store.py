"""Per-user query history and routing preferences."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

HISTORY_LIMIT = 100
EXPIRY_SECONDS = 1800  # 30 min inactivity


@dataclass
class QueryRecord:
    """Single query history entry."""

    query_preview: str  # first 100 chars
    specialty: str
    model: str
    success: bool
    complexity: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserMemory:
    """Per-user routing memory."""

    user_id: str
    query_history: list[QueryRecord] = field(default_factory=list)
    model_accuracy: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    last_active: float = field(default_factory=time.time)

    def record_query(
        self,
        query: str,
        specialty: str,
        model: str,
        success: bool,
        complexity: float,
    ) -> None:
        self.last_active = time.time()
        record = QueryRecord(
            query_preview=query[:100],
            specialty=specialty,
            model=model,
            success=success,
            complexity=complexity,
        )
        self.query_history.append(record)
        if len(self.query_history) > HISTORY_LIMIT:
            self.query_history = self.query_history[-HISTORY_LIMIT:]

        # Track per-model accuracy (1.0 for success, 0.0 for failure)
        self.model_accuracy[model].append(1.0 if success else 0.0)
        # Keep only last 50 per model
        if len(self.model_accuracy[model]) > 50:
            self.model_accuracy[model] = self.model_accuracy[model][-50:]

    @property
    def dominant_specialty(self) -> Optional[str]:
        """Auto-detect the user's dominant specialty from history."""
        if not self.query_history:
            return None
        counts: dict[str, int] = defaultdict(int)
        for r in self.query_history[-30:]:  # last 30 queries
            counts[r.specialty] += 1
        if not counts:
            return None
        top = max(counts, key=counts.get)  # type: ignore
        # Only return if > 40% of recent queries
        if counts[top] / len(self.query_history[-30:]) > 0.4:
            return top
        return None

    @property
    def avg_complexity(self) -> float:
        """Running average complexity of user's queries."""
        if not self.query_history:
            return 0.5
        recent = self.query_history[-20:]
        return sum(r.complexity for r in recent) / len(recent)

    def get_model_avg_accuracy(self, model: str) -> Optional[float]:
        """Get running average accuracy for a specific model."""
        scores = self.model_accuracy.get(model)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def specialty_boost(self, ambiguous_specialty: str) -> float:
        """Return a boost score if ambiguous query matches user's dominant specialty.

        Returns 0.0 to 0.3 boost.
        """
        dom = self.dominant_specialty
        if dom and dom == ambiguous_specialty:
            return 0.2
        return 0.0

    def model_preference_score(self, model: str) -> float:
        """Score based on user's past accuracy with this model.

        Returns -0.1 to +0.1 adjustment.
        """
        avg = self.get_model_avg_accuracy(model)
        if avg is None:
            return 0.0
        # Center around 0.7 (expected baseline accuracy)
        return (avg - 0.7) * 0.3  # max +0.09 / -0.21

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > EXPIRY_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "dominant_specialty": self.dominant_specialty,
            "avg_complexity": round(self.avg_complexity, 2),
            "total_queries": len(self.query_history),
            "model_accuracy": {
                k: round(sum(v) / len(v), 3) if v else 0.0
                for k, v in self.model_accuracy.items()
            },
            "last_active": self.last_active,
            "recent_specialties": [
                r.specialty for r in self.query_history[-10:]
            ],
        }


class UserMemoryStore:
    """In-memory store for per-user memories, keyed by user_id."""

    def __init__(self) -> None:
        self._users: dict[str, UserMemory] = {}

    def get_or_create(self, user_id: str) -> UserMemory:
        if user_id not in self._users:
            self._users[user_id] = UserMemory(user_id=user_id)
        mem = self._users[user_id]
        mem.last_active = time.time()
        return mem

    def get(self, user_id: str) -> Optional[UserMemory]:
        return self._users.get(user_id)

    def list_users(self) -> list[str]:
        return list(self._users.keys())

    def cleanup_expired(self) -> int:
        """Remove expired user memories. Returns count removed."""
        expired = [uid for uid, mem in self._users.items() if mem.is_expired()]
        for uid in expired:
            del self._users[uid]
        return len(expired)


# Singleton
user_memory_store = UserMemoryStore()
