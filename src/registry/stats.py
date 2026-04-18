"""EMA runtime stats + auto-disable/recover."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

EMA_ALPHA = 0.1  # smoothing factor for exponential moving average
LATENCY_DISABLE_FACTOR = 5.0  # disable if latency > 5x expected
ERROR_RATE_THRESHOLD = 0.20  # disable if error rate > 20%
RECOVERY_INTERVAL_S = 30  # check recovery every 30 seconds
RECOVERY_COOL_S = 120  # model must be disabled for 2 min before re-check


@dataclass
class ModelRuntimeStats:
    """Per-model runtime statistics with EMA smoothing."""

    model_name: str
    latency_ema: float = 0.0  # ms
    error_rate_ema: float = 0.0  # 0-1
    total_requests: int = 0
    total_errors: int = 0
    per_specialty_accuracy: dict[str, float] = field(default_factory=dict)
    per_specialty_count: dict[str, int] = field(default_factory=dict)
    disabled_at: Optional[float] = None
    disable_reason: Optional[str] = None
    last_update: float = field(default_factory=time.time)
    latency_history: list[float] = field(default_factory=list)  # last 50 for sparkline
    _degradation_until: float = 0.0  # timestamp when degradation ends
    _degradation_latency: float = 0.0  # injected latency during degradation

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "latency_ema": round(self.latency_ema, 1),
            "error_rate_ema": round(self.error_rate_ema, 4),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "per_specialty_accuracy": {
                k: round(v, 3) for k, v in self.per_specialty_accuracy.items()
            },
            "per_specialty_count": self.per_specialty_count,
            "disabled_at": self.disabled_at,
            "disable_reason": self.disable_reason,
            "latency_history": self.latency_history[-50:],
            "degraded": self._degradation_until > time.time(),
            "degradation_remaining_s": max(0, round(self._degradation_until - time.time())),
        }


class StatsTracker:
    """Track runtime stats for all models with auto-disable/recover."""

    def __init__(self) -> None:
        self._stats: dict[str, ModelRuntimeStats] = {}
        self._expected_latency: dict[str, float] = {}  # from config
        self._recovery_task: Optional[asyncio.Task] = None

    def set_expected_latency(self, model_name: str, latency_ms: float) -> None:
        self._expected_latency[model_name] = latency_ms

    def _get_or_create(self, model_name: str) -> ModelRuntimeStats:
        if model_name not in self._stats:
            self._stats[model_name] = ModelRuntimeStats(model_name=model_name)
        return self._stats[model_name]

    def update_stats(
        self,
        model_name: str,
        latency_ms: float,
        success: bool,
        specialty: Optional[str] = None,
        accuracy: Optional[float] = None,
    ) -> dict[str, Any]:
        """Update stats for a model after a request. Returns action taken."""
        s = self._get_or_create(model_name)
        s.total_requests += 1
        s.last_update = time.time()

        # Inject simulated latency during degradation window
        if s._degradation_until > time.time():
            latency_ms = max(latency_ms, s._degradation_latency)

        # EMA latency
        if s.latency_ema == 0:
            s.latency_ema = latency_ms
        else:
            s.latency_ema = EMA_ALPHA * latency_ms + (1 - EMA_ALPHA) * s.latency_ema

        # Latency sparkline
        s.latency_history.append(round(latency_ms, 1))
        if len(s.latency_history) > 50:
            s.latency_history = s.latency_history[-50:]

        # EMA error rate
        error_val = 0.0 if success else 1.0
        s.error_rate_ema = EMA_ALPHA * error_val + (1 - EMA_ALPHA) * s.error_rate_ema
        if not success:
            s.total_errors += 1

        # Per-specialty accuracy
        if specialty and accuracy is not None:
            if specialty not in s.per_specialty_accuracy:
                s.per_specialty_accuracy[specialty] = accuracy
                s.per_specialty_count[specialty] = 1
            else:
                count = s.per_specialty_count[specialty]
                old_avg = s.per_specialty_accuracy[specialty]
                new_avg = (old_avg * count + accuracy) / (count + 1)
                s.per_specialty_accuracy[specialty] = new_avg
                s.per_specialty_count[specialty] = count + 1

        # Auto-disable check
        action = {"action": "none"}
        expected = self._expected_latency.get(model_name, 1000.0)

        if s.latency_ema > expected * LATENCY_DISABLE_FACTOR and s.total_requests >= 5:
            s.disabled_at = time.time()
            s.disable_reason = (
                f"latency {s.latency_ema:.0f}ms > {LATENCY_DISABLE_FACTOR}x "
                f"expected {expected:.0f}ms"
            )
            action = {"action": "disabled", "reason": s.disable_reason}
            logger.warning("Auto-disabled %s: %s", model_name, s.disable_reason)

        elif s.error_rate_ema > ERROR_RATE_THRESHOLD and s.total_requests >= 5:
            s.disabled_at = time.time()
            s.disable_reason = f"error rate {s.error_rate_ema:.1%} > {ERROR_RATE_THRESHOLD:.0%}"
            action = {"action": "disabled", "reason": s.disable_reason}
            logger.warning("Auto-disabled %s: %s", model_name, s.disable_reason)

        return action

    def get_stats(self, model_name: str) -> Optional[ModelRuntimeStats]:
        return self._stats.get(model_name)

    def all_stats(self) -> list[ModelRuntimeStats]:
        return list(self._stats.values())

    def is_disabled(self, model_name: str) -> bool:
        s = self._stats.get(model_name)
        return s is not None and s.disabled_at is not None

    def performance_score(self, model_name: str) -> float:
        """Compute weighted performance score: accuracy*0.4 + latency*0.3 + cost*0.3.

        Returns 0.5 if no stats available (neutral).
        """
        s = self._stats.get(model_name)
        if s is None or s.total_requests == 0:
            return 0.5

        # Accuracy component (average across specialties, or 0.5 if none)
        if s.per_specialty_accuracy:
            acc = sum(s.per_specialty_accuracy.values()) / len(s.per_specialty_accuracy)
        else:
            acc = 0.5

        # Latency component: normalized 0-1 (lower is better)
        expected = self._expected_latency.get(model_name, 1000.0)
        latency_norm = max(0.0, 1.0 - s.latency_ema / (expected * 3))

        # Error rate component: 1 - error_rate
        reliability = 1.0 - s.error_rate_ema

        return acc * 0.4 + latency_norm * 0.3 + reliability * 0.3

    def simulate_degradation(
        self, model_name: str, latency_spike_ms: float = 5000.0, duration_s: float = 60.0
    ) -> dict:
        """Simulate model latency degradation for demo.

        Injects high-latency (but successful) requests to shift EMA upward.
        Model stays enabled but router sees it as slow → prefers alternatives.
        After duration_s, latency naturally recovers via EMA decay on normal requests.
        """
        s = self._get_or_create(model_name)
        # Inject slow but SUCCESSFUL requests — simulates real-world latency spike
        for _ in range(10):
            self.update_stats(model_name, latency_spike_ms, success=True)

        # Set a degradation window — model is artificially slow for duration_s
        s._degradation_until = time.time() + duration_s
        s._degradation_latency = latency_spike_ms

        return {
            "model": model_name,
            "simulated_latency": latency_spike_ms,
            "duration_s": duration_s,
            "new_ema_latency": round(s.latency_ema, 1),
            "new_error_rate": round(s.error_rate_ema, 4),
            "disabled": s.disabled_at is not None,
            "note": f"Model will appear slow for {duration_s}s. EMA will recover naturally after.",
        }

    async def recovery_check_loop(
        self, registry: Any, interval: float = RECOVERY_INTERVAL_S
    ) -> None:
        """Periodically re-enable models that have recovered.

        Runs as a background task. Resets stats for models that have been
        disabled for at least RECOVERY_COOL_S seconds.
        """
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            for s in self._stats.values():
                if s.disabled_at is None:
                    continue
                if now - s.disabled_at < RECOVERY_COOL_S:
                    continue
                # Reset and re-enable
                s.disabled_at = None
                s.disable_reason = None
                s.latency_ema = self._expected_latency.get(s.model_name, 500.0)
                s.error_rate_ema = 0.0
                # Re-enable in registry
                if registry is not None:
                    registry.update_model(s.model_name, enabled=True)
                logger.info("Recovered model %s after cooldown", s.model_name)

    def start_recovery_loop(self, registry: Any) -> asyncio.Task:
        """Start the recovery check loop as a background task."""
        self._recovery_task = asyncio.create_task(self.recovery_check_loop(registry))
        return self._recovery_task


# Singleton
stats_tracker = StatsTracker()
