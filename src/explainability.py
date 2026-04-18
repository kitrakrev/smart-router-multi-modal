"""Routing explainability — human-readable decision trace.

Produces a structured explanation of WHY a particular model was selected,
covering each signal's contribution to the final decision.

Usage:
    from src.explainability import explain_decision
    explanation = explain_decision(routing_decision)
    print(explanation.text)          # Human-readable
    print(explanation.to_dict())     # JSON-serializable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.router import RoutingDecision


@dataclass
class SignalExplanation:
    """Explanation of a single signal's contribution."""

    signal_name: str
    value: str
    contribution: str  # how it affected routing
    impact: str  # "high", "medium", "low", "none"

    def to_dict(self) -> dict[str, str]:
        return {
            "signal": self.signal_name,
            "value": self.value,
            "contribution": self.contribution,
            "impact": self.impact,
        }


@dataclass
class RoutingExplanation:
    """Full routing explanation."""

    summary: str
    model_selected: str
    specialty: str
    strategy: str
    signal_explanations: list[SignalExplanation] = field(default_factory=list)
    alternatives_considered: list[dict[str, Any]] = field(default_factory=list)
    cost_breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Human-readable explanation."""
        lines = [
            f"=== Routing Explanation ===",
            f"",
            f"Summary: {self.summary}",
            f"Model: {self.model_selected}",
            f"Specialty: {self.specialty}",
            f"Strategy: {self.strategy}",
            f"",
            f"--- Signal Analysis ---",
        ]

        for se in self.signal_explanations:
            icon = {"high": "●", "medium": "◐", "low": "○", "none": "·"}.get(se.impact, "?")
            lines.append(f"  {icon} {se.signal_name}: {se.value}")
            lines.append(f"    → {se.contribution}")

        if self.cost_breakdown:
            lines.append(f"")
            lines.append(f"--- Cost ---")
            lines.append(f"  Estimated: ${self.cost_breakdown.get('estimated', 0):.4f}")
            lines.append(f"  vs GPT-4o: ${self.cost_breakdown.get('gpt4o_estimate', 0):.4f}")
            savings = self.cost_breakdown.get("savings_pct", 0)
            if savings > 0:
                lines.append(f"  Savings: {savings:.0f}%")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "model_selected": self.model_selected,
            "specialty": self.specialty,
            "strategy": self.strategy,
            "signals": [se.to_dict() for se in self.signal_explanations],
            "alternatives": self.alternatives_considered,
            "cost": self.cost_breakdown,
        }


def explain_decision(decision: RoutingDecision) -> RoutingExplanation:
    """Generate a human-readable explanation of a routing decision."""
    signals = decision.signals
    explanations: list[SignalExplanation] = []

    # Text/specialty signal
    text_sig = signals.get("text", {})
    specialty = text_sig.get("matched_specialty", "unknown")
    similarity = text_sig.get("similarity", 0)
    explanations.append(SignalExplanation(
        signal_name="Text Specialty",
        value=f"{specialty} (sim={similarity:.3f})",
        contribution=_text_contribution(specialty, similarity),
        impact="high" if similarity > 0.5 else "medium",
    ))

    # Vision signal
    vision_sig = signals.get("vision", {})
    image_type = vision_sig.get("image_type", "none")
    vision_score = vision_sig.get("similarity_score", 0)
    if image_type != "none":
        explanations.append(SignalExplanation(
            signal_name="Vision",
            value=f"{image_type} (sim={vision_score:.3f})",
            contribution=f"Image classified as {image_type} → routed to vision-capable model",
            impact="high",
        ))
    else:
        explanations.append(SignalExplanation(
            signal_name="Vision",
            value="No image detected",
            contribution="Text-only routing, no vision capability required",
            impact="none",
        ))

    # Complexity signal
    complexity_sig = signals.get("complexity", {})
    complexity = complexity_sig.get("score", 0)
    level = complexity_sig.get("level", "unknown")
    explanations.append(SignalExplanation(
        signal_name="Complexity",
        value=f"{level} ({complexity:.3f})",
        contribution=_complexity_contribution(complexity, level, decision.model_type),
        impact="high" if complexity > 0.7 else "medium" if complexity > 0.3 else "low",
    ))

    # Safety signal
    safety_sig = signals.get("safety", {})
    is_safe = safety_sig.get("is_safe", True)
    risk_score = safety_sig.get("risk_score", 0)
    flags = safety_sig.get("flags", [])
    if not is_safe or risk_score > 0.3:
        explanations.append(SignalExplanation(
            signal_name="Safety",
            value=f"risk={risk_score:.3f}, flags={flags}",
            contribution="Safety concern detected — query may be blocked or flagged",
            impact="high",
        ))
    else:
        explanations.append(SignalExplanation(
            signal_name="Safety",
            value=f"safe (risk={risk_score:.3f})",
            contribution="No safety concerns",
            impact="none",
        ))

    # Tools signal
    tools_sig = signals.get("tools", {})
    needs_tools = tools_sig.get("needs_tools", False)
    recommended = tools_sig.get("recommended", [])
    if needs_tools:
        explanations.append(SignalExplanation(
            signal_name="Tools",
            value=f"needed: {recommended}",
            contribution=f"Tool-capable model preferred for {', '.join(recommended)}",
            impact="medium",
        ))
    else:
        explanations.append(SignalExplanation(
            signal_name="Tools",
            value="not needed",
            contribution="No tools required",
            impact="none",
        ))

    # Modality signal
    modality_sig = signals.get("modality", {})
    modality_type = modality_sig.get("type", "text_only")
    explanations.append(SignalExplanation(
        signal_name="Modality",
        value=modality_type,
        contribution=_modality_contribution(modality_type),
        impact="high" if modality_type != "text_only" else "low",
    ))

    # Ambiguity signal
    ambiguity_sig = signals.get("ambiguity", {})
    if ambiguity_sig.get("is_ambiguous"):
        explanations.append(SignalExplanation(
            signal_name="Ambiguity",
            value=f"LOW CONFIDENCE ({ambiguity_sig.get('confidence', 0):.3f})",
            contribution=ambiguity_sig.get("reason", "Query is ambiguous") + " — " + ambiguity_sig.get("fallback", ""),
            impact="high",
        ))

    # Cost breakdown
    est_cost = decision.estimated_cost
    # GPT-4o cost estimate (2.50/1K input + 10.00/1K output)
    gpt4o_cost = est_cost * 10 if est_cost > 0 else 0.01  # rough
    savings_pct = ((gpt4o_cost - est_cost) / gpt4o_cost * 100) if gpt4o_cost > 0 else 100
    cost_breakdown = {
        "estimated": est_cost,
        "gpt4o_estimate": gpt4o_cost,
        "savings_pct": savings_pct,
        "model_cost_input": 0,
        "model_cost_output": 0,
    }

    # Summary
    summary = _build_summary(decision, signals)

    return RoutingExplanation(
        summary=summary,
        model_selected=decision.model_name,
        specialty=decision.specialty,
        strategy=decision.budget_strategy,
        signal_explanations=explanations,
        cost_breakdown=cost_breakdown,
    )


def _text_contribution(specialty: str, similarity: float) -> str:
    if similarity > 0.7:
        return f"Strong match to {specialty} exemplars — high confidence routing"
    if similarity > 0.5:
        return f"Moderate match to {specialty} — routing with reasonable confidence"
    if similarity > 0.3:
        return f"Weak match to {specialty} — best available but low confidence"
    return f"No strong specialty match — defaulting to {specialty}"


def _complexity_contribution(score: float, level: str, model_type: str) -> str:
    if score > 0.7:
        if model_type == "reasoning":
            return "Complex query → reasoning model selected with extended token budget"
        return "Complex query detected but no reasoning model available, using best capable"
    if score > 0.3:
        return "Medium complexity — standard model with default parameters"
    return "Simple query — lightweight model preferred for cost efficiency"


def _modality_contribution(modality: str) -> str:
    if modality == "multimodal":
        return "Image + text detected → vision-capable model required"
    if modality == "vision":
        return "Image-only input → vision model required"
    return "Text-only → no vision capability needed"


def _build_summary(decision: RoutingDecision, signals: dict) -> str:
    parts = []

    if decision.blocked:
        return f"Query BLOCKED: {decision.block_reason}"

    specialty = decision.specialty
    is_medical = specialty.startswith("medical.")
    spec_name = specialty.split(".")[-1] if "." in specialty else specialty

    if is_medical:
        parts.append(f"Medical query classified as {spec_name}")
    else:
        parts.append(f"General query classified as {spec_name}")

    modality = signals.get("modality", {}).get("type", "text_only")
    if modality != "text_only":
        parts.append(f"with {modality} input")

    complexity = signals.get("complexity", {}).get("level", "medium")
    parts.append(f"({complexity} complexity)")

    parts.append(f"→ routed to {decision.model_name}")

    if decision.budget_strategy == "critical":
        parts.append("using CRITICAL strategy (highest-quality model, extended reasoning budget of 4096 tokens, latency constraints ignored)")
    else:
        parts.append(f"using {decision.budget_strategy} strategy")

    if decision.estimated_cost == 0:
        parts.append("(free, local model)")
    else:
        parts.append(f"(est. ${decision.estimated_cost:.4f})")

    return " ".join(parts)
