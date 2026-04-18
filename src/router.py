"""Decision engine and model selection.

Takes signal results and applies configurable routing rules to select
the optimal model and inference configuration.  Integrates with the
dynamic ModelRegistry for capability-based routing and fallback.

Supports three rule formats:
  - Legacy: rules specify ``model`` directly (hardcoded routing).
  - Budget-aware: rules specify ``require`` (capabilities) + ``strategy``
    and the router picks the cheapest/best model dynamically.
  - Embedding-based (vLLM-SR style): decisions define ``exemplars`` and
    queries are matched via cosine similarity of sentence embeddings.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import yaml

try:
    from src.signals import SignalResult
except ImportError:
    from signals import SignalResult

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is REQUIRED. Install: pip install sentence-transformers"
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    capabilities: list[str]
    quality_score: float = 0.7  # 0-1 expected quality rating


@dataclass
class BudgetConfig:
    """Budget constraints for cost-optimised routing."""
    max_cost_per_query: float = 0.01  # USD hard cap
    strategy: str = "cheapest_capable"  # default global strategy
    quality_threshold: float = 0.7


@dataclass
class RouterDecision:
    selected_model: str
    inference_config: dict
    estimated_cost: float
    estimated_latency_ms: float
    reason: str
    signals_used: dict
    trace_id: str
    blocked: bool = False
    block_reason: str = ""
    decision_name: str = ""
    similarity: float = 0.0


# ---------------------------------------------------------------------------
# DecisionMatcher — vLLM-SR style embedding matching
# ---------------------------------------------------------------------------

class DecisionMatcher:
    """Matches queries to decisions using embedding similarity of exemplars."""

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.decision_embeddings: dict[str, dict] = {}  # name -> {centroid, config}
        self._loaded = False

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_decisions(self, decisions_config: list[dict]):
        """Pre-compute decision embeddings from exemplars."""
        self._ensure_model()
        self.decision_embeddings = {}
        for decision in decisions_config:
            name = decision.get("name", "unknown")
            exemplar_texts = decision.get("exemplars", [])
            if not exemplar_texts:
                continue
            embeddings = self.model.encode(exemplar_texts)
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.decision_embeddings[name] = {
                "centroid": centroid,
                "config": decision,
            }
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded and len(self.decision_embeddings) > 0

    def match(self, query_text: str) -> tuple[str, float]:
        """Find best matching decision for a query.

        Always returns highest scoring decision — no threshold cutoff,
        no default fallback. Embedding match handles ALL queries.
        """
        if not self.decision_embeddings or not query_text.strip():
            # Pick first decision as absolute fallback (shouldn't happen)
            first = next(iter(self.decision_embeddings), "general")
            return (first, 0.0)

        self._ensure_model()
        query_emb = self.model.encode(query_text)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Score ALL decisions, apply optional min_similarity as soft preference
        scores: dict[str, float] = {}
        all_scores: dict[str, float] = {}
        for name, data in self.decision_embeddings.items():
            sim = float(np.dot(query_emb, data["centroid"]))
            all_scores[name] = sim
            min_sim = data["config"].get("min_similarity", 0.0)
            if sim >= min_sim:
                scores[name] = sim

        # If any decision meets threshold, use it
        if scores:
            best = max(scores, key=scores.get)
            return (best, scores[best])

        # Otherwise pick highest overall (no fallback to "general")
        best = max(all_scores, key=all_scores.get)
        return (best, all_scores[best])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.models: dict[str, ModelConfig] = {}
        self.rules: list[dict] = []
        self.default_rule: dict = {"model": "gpt-4o-mini", "config": {"temperature": 0.7},
                                    "reason": "General query, balanced model",
                                    "require": ["text"], "strategy": "balanced"}
        self.budget: BudgetConfig = BudgetConfig()
        self._registry = None  # set via set_registry()
        self._raw_config: dict = {}  # raw parsed YAML for config API

        # Embedding-based decision matching
        self.decision_matcher = DecisionMatcher()
        self.decisions: dict[str, dict] = {}  # name -> decision config
        self.safety_rules: list[dict] = []

        self.load_config()

    # ------------------------------------------------------------------
    # Config access helpers (for config API)
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return the full current config as a JSON-serialisable dict."""
        return copy.deepcopy(self._raw_config)

    def apply_config(self, cfg: dict):
        """Apply a full configuration dict (same schema as config.yaml)."""
        self._raw_config = copy.deepcopy(cfg)
        self._parse_config(cfg)

    def update_budget(self, budget_update: dict):
        """Merge partial budget updates into the current config."""
        routing = self._raw_config.setdefault("routing", {})
        budget_section = routing.setdefault("budget", {})
        budget_section.update(budget_update)
        self._parse_config(self._raw_config)

    def update_rules(self, new_rules: list[dict]):
        """Replace routing rules (but keep models + budget intact)."""
        routing = self._raw_config.setdefault("routing", {})
        routing["rules"] = new_rules
        self._parse_config(self._raw_config)

    def set_registry(self, registry):
        """Attach a ModelRegistry instance for capability-based routing."""
        self._registry = registry

    # ------------------------------------------------------------------
    # Config loading / parsing
    # ------------------------------------------------------------------

    def load_config(self):
        """Load or reload configuration from YAML file on disk."""
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            cfg = {"models": [], "routing": {"rules": []}}

        self._raw_config = copy.deepcopy(cfg)
        self._parse_config(cfg)

    def _parse_config(self, cfg: dict):
        """Parse a config dict into internal structures."""
        # --- Models ---
        self.models = {}
        for m in cfg.get("models", []):
            mc = ModelConfig(
                name=m["name"], provider=m["provider"],
                cost_per_1k_input=m.get("cost_per_1k_input", 0),
                cost_per_1k_output=m.get("cost_per_1k_output", 0),
                avg_latency_ms=m.get("avg_latency_ms", 500),
                capabilities=m.get("capabilities", []),
                quality_score=m.get("quality_score", 0.7),
            )
            self.models[mc.name] = mc

        # --- Budget ---
        routing = cfg.get("routing", {})
        budget_raw = routing.get("budget", {})
        self.budget = BudgetConfig(
            max_cost_per_query=budget_raw.get("max_cost_per_query", 0.01),
            strategy=budget_raw.get("strategy", "cheapest_capable"),
            quality_threshold=budget_raw.get("quality_threshold", 0.7),
        )

        # --- Decisions (new: embedding-based) ---
        decisions_list = routing.get("decisions", [])
        self.decisions = {}
        for d in decisions_list:
            self.decisions[d.get("name", "unknown")] = d

        # Pre-compute decision embeddings if any decisions have exemplars
        exemplar_decisions = [d for d in decisions_list if d.get("exemplars")]
        if exemplar_decisions:
            self.decision_matcher.load_decisions(exemplar_decisions)

        # --- Safety rules (fast keyword-based) ---
        self.safety_rules = routing.get("safety_rules", [])

        # --- Legacy rules (backward compatible) ---
        self.rules = []
        for rule in routing.get("rules", []):
            if "default" in rule:
                self.default_rule = rule["default"]
                # Ensure default has budget-aware fields
                self.default_rule.setdefault("require", ["text"])
                self.default_rule.setdefault("strategy", "balanced")
            else:
                self.rules.append(rule)

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _signals_dict(self, signals: list[SignalResult]) -> dict[str, SignalResult]:
        return {s.name: s for s in signals}

    def _estimate_cost(self, model_name: str, input_tokens: int = 500, output_tokens: int = 300) -> float:
        model = self.models.get(model_name)
        if not model:
            return 0.0
        cost = (input_tokens / 1000) * model.cost_per_1k_input + \
               (output_tokens / 1000) * model.cost_per_1k_output
        return round(cost, 6)

    def _estimate_latency(self, model_name: str) -> float:
        model = self.models.get(model_name)
        return model.avg_latency_ms if model else 500.0

    def _eval_condition(self, condition: str, sig_map: dict[str, SignalResult]) -> bool:
        """Evaluate a rule condition string against signal results.

        Supports both legacy format (``safety.score > 0.4``) and the new
        shorthand format (``safety > 0.7``, ``vision.detected``, etc.).
        """
        try:
            # Build evaluation context
            ctx: dict[str, Any] = {}

            safety = sig_map.get("safety")
            if safety:
                ctx["safety_score"] = safety.score
                ctx["safety_flagged"] = safety.metadata.get("flagged", False)

            vision = sig_map.get("vision")
            if vision:
                ctx["vision_detected"] = vision.metadata.get("detected", False)

            tool = sig_map.get("tool")
            if tool:
                ctx["tool_needed"] = tool.metadata.get("needed", False)
                ctx["tool_score"] = tool.score

            complexity = sig_map.get("complexity")
            if complexity:
                ctx["complexity_score"] = complexity.score

            domain = sig_map.get("domain")
            if domain:
                ctx["domain_name"] = domain.metadata.get("domain", "general")
                ctx["domain_score"] = domain.score

            modality = sig_map.get("modality")
            if modality:
                ctx["modality"] = modality.metadata.get("modality", "text")

            language = sig_map.get("language")
            if language:
                ctx["language"] = language.metadata.get("detected", "en")

            pii = sig_map.get("pii")
            if pii:
                ctx["pii_score"] = pii.score

            # Evaluate condition with simple string matching
            cond = condition

            # New shorthand: "safety > 0.7" -> "safety.score > 0.7"
            import re
            cond = re.sub(r'\bsafety\s*([><=!])', r'safety.score \1', cond)
            cond = re.sub(r'\bcomplexity\s*([><=!])', r'complexity.score \1', cond)
            cond = re.sub(r'\bpii\s*([><=!])', r'pii.score \1', cond)
            cond = re.sub(r'\bdomain\s+(in\b|==)', r'domain.name \1', cond)

            # Legacy replacements
            cond = cond.replace("safety.score", str(ctx.get("safety_score", 0)))
            cond = cond.replace("vision.detected", str(ctx.get("vision_detected", False)))
            cond = cond.replace("tool.needed", str(ctx.get("tool_needed", False)))
            cond = cond.replace("complexity.score", str(ctx.get("complexity_score", 0)))
            cond = cond.replace("domain.name", repr(ctx.get("domain_name", "general")))
            cond = cond.replace("pii.score", str(ctx.get("pii_score", 0)))

            return bool(eval(cond, {"__builtins__": {}}, {"true": True, "false": False, "True": True, "False": False}))
        except Exception:
            return False

    def _eval_safety_condition(self, condition: str, sig_map: dict[str, SignalResult]) -> bool:
        """Evaluate a safety rule condition. Same logic as _eval_condition."""
        return self._eval_condition(condition, sig_map)

    # ------------------------------------------------------------------
    # Budget-aware model selection
    # ------------------------------------------------------------------

    def _select_model_by_strategy(
        self, required_capabilities: list[str], strategy: str
    ) -> tuple[str | None, dict]:
        """Pick a model from self.models based on capabilities and strategy.

        Returns (model_name, inference_config) or (None, {}) if no model found.
        """
        req_set = set(required_capabilities)

        # Find models that have ALL required capabilities
        capable: list[ModelConfig] = []
        for m in self.models.values():
            model_caps = set(m.capabilities)
            # Accept models whose capabilities contain the required ones.
            # Also accept models with "general" as a wildcard for "text".
            effective_caps = set(model_caps)
            if "general" in effective_caps or "fast" in effective_caps:
                effective_caps.add("text")
            if req_set.issubset(effective_caps):
                capable.append(m)

        if not capable:
            # Fallback: return cheapest model regardless of capabilities
            if self.models:
                fallback = min(self.models.values(), key=lambda m: m.cost_per_1k_output)
                return (fallback.name, {})
            return (None, {})

        # Filter by budget hard cap
        max_cost = self.budget.max_cost_per_query
        affordable = [
            m for m in capable
            if self._estimate_cost(m.name) <= max_cost
        ]
        # If none are affordable, use all capable (don't silently drop)
        pool = affordable if affordable else capable

        if strategy == "performance_weighted":
            # Score based on recent runtime performance
            for model in pool:
                runtime_stats = None
                if self._registry and hasattr(self._registry, 'runtime_stats'):
                    runtime_stats = self._registry.runtime_stats.get(model.name)
                if runtime_stats and runtime_stats.total_requests > 10:
                    latency_score = max(0, 1 - runtime_stats.latency_ema / 5000)
                    accuracy_score = 1 - runtime_stats.error_rate_ema
                    cost_score = max(0, 1 - model.cost_per_1k_output / 20)
                    model._runtime_score = accuracy_score * 0.4 + latency_score * 0.3 + cost_score * 0.3
                else:
                    model._runtime_score = model.quality_score or 0.5
            pick = max(pool, key=lambda m: getattr(m, '_runtime_score', 0.5))
        elif strategy == "cheapest_capable":
            pick = min(pool, key=lambda m: m.cost_per_1k_output)
        elif strategy == "quality_first":
            pick = max(pool, key=lambda m: m.quality_score)
        elif strategy == "balanced":
            # Minimise cost/quality ratio
            pick = min(pool, key=lambda m: m.cost_per_1k_output / max(m.quality_score, 0.1))
        else:
            # Unknown strategy -> cheapest
            pick = min(pool, key=lambda m: m.cost_per_1k_output)

        return (pick.name, {})

    # ------------------------------------------------------------------
    # Capability helpers (used by registry-aware decide)
    # ------------------------------------------------------------------

    def _required_capabilities(self, sig_map: dict[str, SignalResult]) -> set[str]:
        """Derive required model capabilities from signal results."""
        required: set[str] = set()
        vision = sig_map.get("vision")
        if vision and vision.metadata.get("detected"):
            required.add("vision")
        tool = sig_map.get("tool")
        if tool and tool.metadata.get("needed"):
            required.add("tools")
        modality = sig_map.get("modality")
        if modality:
            mod = modality.metadata.get("modality", "text")
            if mod == "code_gen":
                required.add("code")
        complexity = sig_map.get("complexity")
        if complexity and complexity.score > 0.7:
            required.add("reasoning")
        if not required:
            required.add("text")
        return required

    async def _registry_select(self, required_caps: set[str], preferred_model: str | None = None) -> tuple[str | None, str]:
        """Pick a model from the registry that satisfies required_caps.

        Returns (model_name, selection_reason). Falls back to cheapest capable.
        """
        if self._registry is None:
            return (preferred_model, "no registry attached")

        try:
            from src.models import ModelCapability
        except ImportError:
            from models import ModelCapability

        cap_enums = set()
        for c in required_caps:
            try:
                cap_enums.add(ModelCapability(c))
            except ValueError:
                pass
        if not cap_enums:
            cap_enums = {ModelCapability.TEXT}

        # If a preferred model is specified, check it first
        if preferred_model:
            entry = await self._registry.get_model(preferred_model)
            if entry and entry.enabled and cap_enums.issubset(entry.capabilities):
                return (preferred_model, f"preferred model satisfies {required_caps}")

        # Find cheapest capable model
        cheapest = await self._registry.get_cheapest_capable(cap_enums)
        if cheapest:
            return (cheapest.name, f"cheapest model with {required_caps}")

        # Fallback: any enabled model
        all_models = await self._registry.list_models()
        enabled = [m for m in all_models if m.enabled]
        if enabled:
            pick = min(enabled, key=lambda m: m.cost_per_1k_input)
            return (pick.name, f"fallback (no model has all of {required_caps})")

        return (preferred_model, "no models in registry, using rule default")

    def decide(self, signals: list[SignalResult], trace_id: Optional[str] = None,
               query_text: str = "") -> RouterDecision:
        """Apply routing rules to signal results and return a decision.

        Supports three modes:
          1. Safety rules (keyword-based, always checked first)
          2. Embedding-based decisions (if config has ``decisions`` with ``exemplars``)
          3. Legacy rule-based (if config has ``rules`` with ``if:`` conditions)

        Both embedding-based and legacy rules can coexist. Embedding-based
        decisions are preferred when available; legacy rules are used as
        fallback or when no decisions are configured.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        sig_map = self._signals_dict(signals)
        signals_summary = {}
        for s in signals:
            signals_summary[s.name] = {
                "score": s.score,
                "confidence": s.confidence,
                "time_ms": s.execution_time_ms,
                "metadata": s.metadata,
                "skipped": s.skipped,
            }

        # ---------------------------------------------------------------
        # 1. Safety rules first (fast keyword-based check)
        # ---------------------------------------------------------------
        for rule in self.safety_rules:
            condition = rule.get("if", "")
            action = rule.get("action", "")
            if self._eval_safety_condition(condition, sig_map):
                if action == "block":
                    return RouterDecision(
                        selected_model="BLOCKED",
                        inference_config={},
                        estimated_cost=0.0,
                        estimated_latency_ms=0.0,
                        reason=rule.get("reason", f"Blocked by safety rule '{rule.get('name', '?')}'"),
                        signals_used=signals_summary,
                        trace_id=trace_id,
                        blocked=True,
                        block_reason=rule.get("reason", "Safety risk detected"),
                        decision_name=rule.get("name", "safety_block"),
                        similarity=0.0,
                    )
                # "warn" action: log but don't block — continue routing

        # ---------------------------------------------------------------
        # 2. Embedding-based decision matching (vLLM-SR style)
        # ---------------------------------------------------------------
        if self.decision_matcher.is_loaded and query_text.strip():
            decision_name, similarity = self.decision_matcher.match(query_text)
            decision_config = self.decisions.get(decision_name, {})

            if decision_config:
                required = decision_config.get("require", ["text"])
                strategy = decision_config.get("strategy", self.budget.strategy)

                # Complexity-aware: only override strategy for pure text decisions
                # NEVER override requirements — vision/tools/reasoning must be respected
                complexity_sig = sig_map.get("complexity")
                complexity_score = complexity_sig.score if complexity_sig else 0.5

                # Only downgrade strategy if decision requires just [text] and query is simple
                if complexity_score < 0.3 and required == ["text"]:
                    strategy = "cheapest_capable"

                model_name, _ = self._select_model_by_strategy(required, strategy)
                if model_name is None:
                    # No model has exact capabilities — DON'T relax to text-only
                    # Instead pick cheapest model that has ANY of the required capabilities
                    req_set = set(required)
                    partial = [(m, len(set(m.capabilities) & req_set)) for m in self.models.values()]
                    partial = [(m, overlap) for m, overlap in partial if overlap > 0]
                    if partial:
                        # Pick cheapest among models with most capability overlap
                        max_overlap = max(o for _, o in partial)
                        best = [m for m, o in partial if o == max_overlap]
                        model_name = min(best, key=lambda m: m.cost_per_1k_output).name
                    else:
                        # Absolute fallback — no overlap at all
                        cheapest = min(self.models.values(), key=lambda m: m.cost_per_1k_output)
                        model_name = cheapest.name

                inference_config = dict(decision_config.get("config", {}))

                return RouterDecision(
                    selected_model=model_name,
                    inference_config=inference_config,
                    estimated_cost=self._estimate_cost(model_name),
                    estimated_latency_ms=self._estimate_latency(model_name),
                    reason=f"Matched '{decision_name}' (similarity: {similarity:.3f})",
                    signals_used=signals_summary,
                    trace_id=trace_id,
                    decision_name=decision_name,
                    similarity=round(similarity, 4),
                )

            # Matched decision name but no config (e.g. 'general' fallback)
            # Fall through to legacy rules or default

        # ---------------------------------------------------------------
        # 3. Legacy rule-based matching (backward compatible)
        # ---------------------------------------------------------------
        for rule in self.rules:
            condition = rule.get("if", "")
            if self._eval_condition(condition, sig_map):
                # Check if this is a block action
                if rule.get("action") == "block":
                    return RouterDecision(
                        selected_model="BLOCKED",
                        inference_config={},
                        estimated_cost=0.0,
                        estimated_latency_ms=0.0,
                        reason=rule.get("reason", "Blocked by routing rule"),
                        signals_used=signals_summary,
                        trace_id=trace_id,
                        blocked=True,
                        block_reason=rule.get("reason", "Safety risk detected"),
                        decision_name=rule.get("name", "blocked"),
                        similarity=0.0,
                    )

                # --- Budget-aware routing (new format) ---
                if "require" in rule:
                    required_caps = rule["require"]
                    strategy = rule.get("strategy", self.budget.strategy)
                    model_name, config = self._select_model_by_strategy(required_caps, strategy)
                    if model_name is None:
                        model_name = self.default_rule.get("model", "gpt-4o-mini")
                    reason = rule.get("reason",
                                      f"Rule '{rule.get('name', '?')}': "
                                      f"require={required_caps} strategy={strategy} -> {model_name}")
                    return RouterDecision(
                        selected_model=model_name,
                        inference_config=rule.get("config", config),
                        estimated_cost=self._estimate_cost(model_name),
                        estimated_latency_ms=self._estimate_latency(model_name),
                        reason=reason,
                        signals_used=signals_summary,
                        trace_id=trace_id,
                        decision_name=rule.get("name", "rule_match"),
                        similarity=0.0,
                    )

                # --- Legacy routing (model field) ---
                model_name = rule.get("model", self.default_rule.get("model", "gpt-4o-mini"))
                config = rule.get("config", {})
                return RouterDecision(
                    selected_model=model_name,
                    inference_config=config,
                    estimated_cost=self._estimate_cost(model_name),
                    estimated_latency_ms=self._estimate_latency(model_name),
                    reason=rule.get("reason", "Matched routing rule"),
                    signals_used=signals_summary,
                    trace_id=trace_id,
                    decision_name=rule.get("name", "legacy_rule"),
                    similarity=0.0,
                )

        # ---------------------------------------------------------------
        # 4. Default rule -- also supports budget-aware
        # ---------------------------------------------------------------
        if "require" in self.default_rule:
            required_caps = self.default_rule["require"]
            strategy = self.default_rule.get("strategy", self.budget.strategy)
            model_name, config = self._select_model_by_strategy(required_caps, strategy)
            if model_name is None:
                model_name = self.default_rule.get("model", "gpt-4o-mini")
            return RouterDecision(
                selected_model=model_name,
                inference_config=self.default_rule.get("config", config),
                estimated_cost=self._estimate_cost(model_name),
                estimated_latency_ms=self._estimate_latency(model_name),
                reason=self.default_rule.get("reason", "Default routing (budget-aware)"),
                signals_used=signals_summary,
                trace_id=trace_id,
                decision_name="default",
                similarity=0.0,
            )

        model_name = self.default_rule.get("model", "gpt-4o-mini")
        config = self.default_rule.get("config", {"temperature": 0.7})
        return RouterDecision(
            selected_model=model_name,
            inference_config=config,
            estimated_cost=self._estimate_cost(model_name),
            estimated_latency_ms=self._estimate_latency(model_name),
            reason=self.default_rule.get("reason", "Default routing"),
            signals_used=signals_summary,
            trace_id=trace_id,
            decision_name="default",
            similarity=0.0,
        )

    async def decide_with_registry(self, signals: list[SignalResult], trace_id: Optional[str] = None,
                                    query_text: str = "") -> RouterDecision:
        """Like decide(), but enhances model selection via the ModelRegistry.

        1. Run normal rule-based decide() to get a candidate.
        2. Check that the candidate has required capabilities via the registry.
        3. If not, find a suitable replacement.
        """
        decision = self.decide(signals, trace_id, query_text=query_text)
        if decision.blocked or self._registry is None:
            return decision

        sig_map = self._signals_dict(signals)
        required = self._required_capabilities(sig_map)
        model_name, reason = await self._registry_select(required, decision.selected_model)

        if model_name and model_name != decision.selected_model:
            # Update decision with registry-selected model
            decision.selected_model = model_name
            decision.estimated_cost = self._estimate_cost(model_name)
            decision.estimated_latency_ms = self._estimate_latency(model_name)
            decision.reason = f"{decision.reason} [registry: {reason}]"

        return decision
