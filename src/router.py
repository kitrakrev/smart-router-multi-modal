"""MedVisionRouter v2 — Two-level routing with specialty taxonomy.

Flow:
  1. Run all signals in parallel (text, vision, complexity, safety, tools, modality)
  2. Determine specialty from text + vision signals + taxonomy lookup
  3. Select model based on specialty, capabilities, budget strategy, and runtime stats
  4. Build prompt from PromptManager (DSPy > manual > auto)
  5. Return routing decision with full trace

Budget strategies:
  - cheapest_capable: cheapest model that meets all requirements
  - quality_first: highest quality_score model
  - balanced: weighted combination of cost, quality, and latency
  - performance_weighted: use per-specialty accuracy from runtime stats
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.signals import AllSignalsResult, run_all_signals
from src.taxonomy import SpecialtyTree, resolve_alias
from src.registry import model_registry, stats_tracker, ModelEntry
from src.prompts import prompt_manager
from src.memory import session_store, user_memory_store, SessionTrace

logger = logging.getLogger(__name__)

BALANCED_WEIGHTS = {"quality": 0.4, "cost_inv": 0.3, "latency_inv": 0.3}


@dataclass
class RoutingDecision:
    """Complete routing decision with trace info."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    model_id: str = ""
    model_type: str = ""
    specialty: str = ""
    system_prompt: str = ""
    prompt_source: str = ""
    inference_params: dict[str, Any] = field(default_factory=dict)
    budget_strategy: str = "balanced"
    estimated_cost: float = 0.0
    reasoning_tokens: int = 0
    tools_available: list[str] = field(default_factory=list)

    signals: dict[str, Any] = field(default_factory=dict)
    routing_latency_ms: float = 0.0

    blocked: bool = False
    block_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "specialty": self.specialty,
            "system_prompt": self.system_prompt[:200],
            "prompt_source": self.prompt_source,
            "inference_params": self.inference_params,
            "budget_strategy": self.budget_strategy,
            "estimated_cost": round(self.estimated_cost, 6),
            "reasoning_tokens": self.reasoning_tokens,
            "tools_available": self.tools_available,
            "signals": self.signals,
            "routing_latency_ms": round(self.routing_latency_ms, 2),
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


class MedVisionRouter:
    """Two-level medical vision router with budget awareness."""

    def __init__(self, budget_strategy: str = "balanced") -> None:
        self.budget_strategy = budget_strategy
        self._taxonomy: Optional[SpecialtyTree] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy-load taxonomy and models."""
        if self._initialized:
            return
        self._taxonomy = await SpecialtyTree.get_instance()
        self._initialized = True
        logger.info("MedVisionRouter initialized")

    async def route(
        self,
        messages: list[dict],
        image_data: Optional[str] = None,
        user_id: Optional[str] = None,
        budget_strategy: Optional[str] = None,
        session_id: Optional[str] = None,
        max_reasoning_tokens: int = 0,
    ) -> RoutingDecision:
        """Route a request through the full pipeline."""
        await self.initialize()
        t0 = time.perf_counter()
        strategy = budget_strategy or self.budget_strategy
        decision = RoutingDecision(budget_strategy=strategy)

        # ── Step 1: Run all signals in parallel ─────────────────────────
        try:
            signals = await run_all_signals(messages, image_data)
        except Exception as e:
            logger.error("Signal pipeline failed: %s", e)
            decision.blocked = True
            decision.block_reason = f"Signal error: {e}"
            decision.routing_latency_ms = (time.perf_counter() - t0) * 1000
            return decision

        decision.signals = self._pack_signals(signals)

        # ── Step 1b: Auto-detect critical queries ───────────────────────
        auto_critical = False
        emergency_keywords = [
            "emergency", "life-threatening", "code blue", "cardiac arrest",
            "anaphylaxis", "stroke", "hemorrhage", "sepsis", "trauma",
            "respiratory failure", "status epilepticus", "acute mi",
        ]
        query_text = self._extract_query_preview(messages, max_len=500).lower()
        if any(kw in query_text for kw in emergency_keywords):
            auto_critical = True
        if signals.complexity.complexity_score > 0.8:
            auto_critical = True

        if auto_critical and strategy != "critical":
            # Auto-upgrade but allow explicit override
            if budget_strategy is None:
                strategy = "critical"
                decision.budget_strategy = "critical"
                logger.info("Auto-upgraded to critical strategy (emergency/complex)")

        # ── Step 2: Safety gate ──────────────────────────────────────────
        if not signals.safety.is_safe and signals.safety.risk_score > 0.8:
            decision.blocked = True
            decision.block_reason = (
                f"Safety block: {', '.join(signals.safety.flags)} "
                f"(score={signals.safety.risk_score:.2f})"
            )
            decision.routing_latency_ms = (time.perf_counter() - t0) * 1000
            return decision

        # ── Step 3: Determine specialty ──────────────────────────────────
        specialty = await self._resolve_specialty(signals)
        decision.specialty = specialty

        # ── Step 3b: Ambiguity detection + LLM fallback ──────────────────
        ambiguity_threshold = 0.35
        if signals.text.similarity < ambiguity_threshold:
            # Embedding uncertain — call fallback LLM classifier
            fallback_domain = None
            fallback_info = {}
            try:
                from src.fallback_classifier import FallbackClassifier
                classifier = await FallbackClassifier.get_instance()
                query_text = self._extract_query_preview(messages, max_len=500)
                result = await classifier.classify(query_text)
                fallback_domain = result.domain
                fallback_info = {
                    "llm_domain": result.domain,
                    "llm_confidence": result.confidence,
                    "llm_model": result.model_used,
                    "llm_latency_ms": result.latency_ms,
                }
                # Use LLM's classification as the specialty
                if result.confidence > 0.5:
                    if result.domain in ("code", "reasoning", "creative", "simple_qa"):
                        specialty = f"general.{result.domain}"
                    else:
                        specialty = f"medical.{result.domain}"
                    decision.specialty = specialty
                    logger.info(
                        "Ambiguous query resolved by LLM: %s (conf=%.2f, %sms)",
                        result.domain, result.confidence, result.latency_ms,
                    )
            except Exception as e:
                logger.warning("Fallback classifier failed: %s", e)

            decision.signals["ambiguity"] = {
                "is_ambiguous": True,
                "confidence": round(signals.text.similarity, 4),
                "reason": f"Low embedding match ({signals.text.similarity:.3f} < {ambiguity_threshold})",
                "fallback": "LLM classifier" if fallback_domain else "quality_first default",
                **fallback_info,
            }
            # Upgrade strategy for ambiguous queries
            if strategy not in ("critical", "quality_first"):
                strategy = "quality_first"
                decision.budget_strategy = "quality_first"
        else:
            decision.signals["ambiguity"] = {
                "is_ambiguous": False,
                "confidence": round(signals.text.similarity, 4),
            }

        # ── Step 4: Determine required capabilities ──────────────────────
        required_caps = self._determine_capabilities(signals, specialty)

        # ── Step 5: Select model by strategy ─────────────────────────────
        model = self._select_model(required_caps, specialty, strategy, user_id=user_id)
        if model is None:
            model = self._select_model_partial(required_caps, specialty)

        if model is None:
            all_models = model_registry.list_models()
            enabled = [m for m in all_models if m.enabled and m.approved]
            if enabled:
                model = min(enabled, key=lambda m: m.cost_per_1k_input)
                logger.warning("No capable model, using fallback: %s", model.name)
            else:
                decision.blocked = True
                decision.block_reason = "No models available"
                decision.routing_latency_ms = (time.perf_counter() - t0) * 1000
                return decision

        decision.model_name = model.name
        decision.model_id = model.model_id
        decision.model_type = model.type

        # ── Step 6: Get prompt for model x specialty ─────────────────────
        prompt_result = prompt_manager.get_prompt(
            model_name=model.name,
            model_type=model.type,
            specialty=specialty,
        )
        decision.system_prompt = prompt_result.system_prompt
        decision.prompt_source = prompt_result.source
        decision.inference_params = dict(prompt_result.params)

        # ── Step 7: Reasoning token budget ───────────────────────────────
        if strategy == "critical":
            # Critical strategy always allocates extended reasoning budget
            decision.reasoning_tokens = max(max_reasoning_tokens, 4096)
            decision.inference_params["max_tokens"] = decision.reasoning_tokens
        elif max_reasoning_tokens > 0:
            decision.reasoning_tokens = max_reasoning_tokens
            decision.inference_params["max_tokens"] = max_reasoning_tokens
        elif signals.complexity.complexity_score > 0.7 and model.type == "reasoning":
            decision.reasoning_tokens = 2048
            decision.inference_params.setdefault("max_tokens", 2048)

        # ── Step 8: Tool awareness ───────────────────────────────────────
        if signals.tools.needs_tools:
            decision.tools_available = signals.tools.recommended_tools

        # ── Step 9: Cost estimation ──────────────────────────────────────
        est_input_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        est_output_tokens = decision.inference_params.get("max_tokens", 1024)
        decision.estimated_cost = (
            (est_input_tokens / 1000) * model.cost_per_1k_input
            + (est_output_tokens / 1000) * model.cost_per_1k_output
        )

        decision.routing_latency_ms = (time.perf_counter() - t0) * 1000

        # ── Step 10: Record trace ────────────────────────────────────────
        if session_id:
            trace = SessionTrace(
                trace_id=decision.trace_id,
                query_preview=self._extract_query_preview(messages),
                has_image=signals.modality.has_image,
                specialty_matched=specialty,
                specialty_similarity=signals.text.similarity,
                image_type_detected=signals.vision.image_type,
                complexity_score=signals.complexity.complexity_score,
                safety_score=signals.safety.risk_score,
                tools_suggested=signals.tools.recommended_tools,
                model_selected=model.name,
                model_type=model.type,
                prompt_used=prompt_result.source,
                reasoning_tokens=decision.reasoning_tokens,
                cost_estimate=decision.estimated_cost,
                routing_latency_ms=decision.routing_latency_ms,
            )
            session = session_store.get_or_create(session_id, user_id or "")
            session.add_trace(trace)

        return decision

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _resolve_specialty(self, signals: AllSignalsResult) -> str:
        """Resolve specialty from text and vision signals + taxonomy."""
        text_specialty = signals.text.matched_specialty

        # If vision detected a specific image type with HIGH confidence, use it
        # Threshold 0.5 avoids routing non-medical images (e.g. Pikachu) to medical
        vision_threshold = 0.5
        if (
            signals.vision.image_type not in ("none", "unknown")
            and signals.vision.similarity_score > vision_threshold
            and self._taxonomy
        ):
            image_type = signals.vision.image_type
            image_specialties = self._taxonomy.get_specialties_for_image_type(image_type)
            if image_specialties:
                # Prefer specialty that matches both image type and text signal
                if text_specialty.startswith("medical."):
                    text_base = text_specialty.split(".")[-1]
                    for sp in image_specialties:
                        if text_base in sp.path:
                            return sp.path
                return image_specialties[0].path

        # If image present but low vision confidence AND text isn't medical,
        # treat as general query with image (not medical)
        if (
            signals.modality.has_image
            and signals.vision.similarity_score <= vision_threshold
            and not text_specialty.startswith("medical.")
        ):
            return text_specialty  # e.g. general.simple_qa for "what is this image?"

        # Resolve through taxonomy for dedup
        if text_specialty.startswith("medical.") and self._taxonomy:
            parts = text_specialty.split(".")
            if len(parts) >= 2:
                resolved = self._taxonomy.resolve(parts[1])
                if resolved:
                    return resolved.path

        return text_specialty

    def _determine_capabilities(
        self, signals: AllSignalsResult, specialty: str
    ) -> list[str]:
        """Determine required model capabilities from signals."""
        caps = ["text"]

        if signals.modality.has_image:
            caps.append("vision")

        # Only require medical capability if confident about medical content
        is_confident_medical = (
            specialty.startswith("medical.")
            and signals.text.similarity > 0.3
        )
        # For images: also check vision confidence
        if signals.modality.has_image and signals.vision.similarity_score < 0.5:
            is_confident_medical = False  # non-medical image overrides text

        if is_confident_medical:
            caps.append("medical")

        if signals.complexity.complexity_score > 0.7:
            caps.append("reasoning")

        if signals.tools.needs_tools:
            caps.append("tools")

        return caps

    def _select_model(
        self,
        required_caps: list[str],
        specialty: str,
        strategy: str,
        user_id: Optional[str] = None,
    ) -> Optional[ModelEntry]:
        """Select model using budget strategy + user/global feedback.

        Scoring incorporates:
        - Base quality_score from config
        - Global reputation: per-specialty accuracy from all users (EMA stats)
        - User-level preference: per-user model accuracy history
        - Critical strategy bypasses all feedback — always picks best base quality
        """
        capable = model_registry.get_capable_models(required_caps)

        # Filter by specialty for medical queries
        if specialty.startswith("medical."):
            spec_name = specialty.split(".")[1] if "." in specialty else specialty
            specialty_models = [
                m for m in capable
                if spec_name in m.specialties or "medical" in m.capabilities
            ]
            if specialty_models:
                capable = specialty_models

        # Filter out stats-disabled models
        capable = [
            m for m in capable
            if not (stats_tracker.get_stats(m.name) and stats_tracker.get_stats(m.name).disabled_at)
        ]

        if not capable:
            return None

        # CRITICAL: always best model, ignore all feedback/metrics
        if strategy == "critical":
            return max(capable, key=lambda m: m.quality_score)

        if strategy == "cheapest_capable":
            return min(capable, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

        if strategy == "quality_first":
            return max(capable, key=lambda m: m.quality_score)

        if strategy == "performance_weighted":
            spec_key = specialty.split(".")[-1] if "." in specialty else specialty

            def perf_score(m: ModelEntry) -> float:
                # Global reputation from all users
                global_score = m.quality_score
                s = stats_tracker.get_stats(m.name)
                if s and spec_key in s.per_specialty_accuracy:
                    global_score = s.per_specialty_accuracy[spec_key]

                # User-level adjustment (single user's experience)
                user_adj = 0.0
                if user_id:
                    user_mem = user_memory_store.get(user_id)
                    if user_mem:
                        user_adj = user_mem.model_preference_score(m.name)

                return global_score + user_adj

            return max(capable, key=perf_score)

        # balanced (default) — incorporates user + global feedback
        def balanced_score(m: ModelEntry) -> float:
            cost = m.cost_per_1k_input + m.cost_per_1k_output
            max_cost = max(
                (c.cost_per_1k_input + c.cost_per_1k_output for c in capable), default=1.0
            ) or 1.0
            max_latency = max((c.avg_latency_ms for c in capable), default=1.0) or 1.0

            cost_inv = 1.0 - (cost / max_cost) if max_cost > 0 else 1.0
            latency_inv = 1.0 - (m.avg_latency_ms / max_latency) if max_latency > 0 else 1.0

            base = (
                BALANCED_WEIGHTS["quality"] * m.quality_score
                + BALANCED_WEIGHTS["cost_inv"] * cost_inv
                + BALANCED_WEIGHTS["latency_inv"] * latency_inv
            )

            # Global reputation adjustment: if many users gave bad feedback,
            # global per-specialty accuracy drops → model score drops for everyone
            global_adj = 0.0
            s = stats_tracker.get_stats(m.name)
            if s and s.total_requests >= 5:
                global_perf = stats_tracker.performance_score(m.name)
                # Shift from neutral (0.5) — positive if good, negative if bad
                global_adj = (global_perf - 0.5) * 0.2  # max ±0.1

            # Per-user adjustment: only affects this user's routing
            # One user's bad experience doesn't penalize globally
            user_adj = 0.0
            if user_id:
                user_mem = user_memory_store.get(user_id)
                if user_mem:
                    user_adj = user_mem.model_preference_score(m.name)
                    # max +0.09 (good history) / -0.21 (bad history)

            return base + global_adj + user_adj

        return max(capable, key=balanced_score)

    def _select_model_partial(
        self,
        required_caps: list[str],
        specialty: str,
    ) -> Optional[ModelEntry]:
        """Fallback: find model with most capability overlap."""
        all_models = model_registry.list_models()
        enabled = [m for m in all_models if m.enabled and m.approved]
        if not enabled:
            return None

        req_set = set(required_caps)
        scored = [(len(req_set & set(m.capabilities)), m.quality_score, m) for m in enabled]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    def _pack_signals(self, signals: AllSignalsResult) -> dict[str, Any]:
        """Pack signal results into serializable dict."""
        return {
            "text": {
                "matched_specialty": signals.text.matched_specialty,
                "similarity": round(signals.text.similarity, 4),
                "is_medical": signals.text.is_medical,
                "top_scores": dict(
                    sorted(signals.text.all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                ),
            },
            "vision": {
                "image_type": signals.vision.image_type,
                "similarity_score": round(signals.vision.similarity_score, 4),
                "top_scores": dict(
                    sorted(signals.vision.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                ) if signals.vision.all_scores else {},
            },
            "complexity": {
                "score": round(signals.complexity.complexity_score, 4),
                "level": signals.complexity.label,
            },
            "safety": {
                "is_safe": signals.safety.is_safe,
                "risk_score": round(signals.safety.risk_score, 4),
                "flags": signals.safety.flags,
            },
            "tools": {
                "needs_tools": signals.tools.needs_tools,
                "recommended": signals.tools.recommended_tools,
                "top_score": round(signals.tools.top_score, 4),
            },
            "modality": {
                "type": signals.modality.modality,
                "has_image": signals.modality.has_image,
                "image_count": signals.modality.image_count,
            },
        }

    @staticmethod
    def _extract_query_preview(messages: list[dict], max_len: int = 100) -> str:
        """Extract text preview from the last user message."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:max_len]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")[:max_len]
        return ""


# Singleton router
router = MedVisionRouter()
