"""Probe new models to discover capabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of probing a model for a single specialty."""

    specialty: str
    question: str
    response: str
    expected_keywords: list[str]
    matched_keywords: list[str]
    accuracy: float  # fraction of expected keywords found

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialty": self.specialty,
            "question": self.question,
            "response_preview": self.response[:200],
            "expected_keywords": self.expected_keywords,
            "matched_keywords": self.matched_keywords,
            "accuracy": self.accuracy,
        }


@dataclass
class ModelProbeReport:
    """Full probe report for a model across specialties."""

    model_name: str
    results: list[ProbeResult] = field(default_factory=list)
    specialty_scores: dict[str, float] = field(default_factory=dict)
    approved: bool = False  # needs human approval via dashboard

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "specialty_scores": self.specialty_scores,
            "approved": self.approved,
            "results": [r.to_dict() for r in self.results],
        }


class CapabilityProber:
    """Probe models to discover capabilities using config/probes.yaml."""

    def __init__(self) -> None:
        self._probes: dict[str, Any] = {}
        self._reports: dict[str, ModelProbeReport] = {}

    def load_probes(self, config_path: str | Path) -> None:
        """Load probe questions from YAML config."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Probes config not found: %s", path)
            return

        with open(path) as f:
            cfg = yaml.safe_load(f)

        self._probes = cfg.get("probes", {})
        logger.info("Loaded probes for categories: %s", list(self._probes.keys()))

    def get_probe_questions(self) -> list[dict[str, Any]]:
        """Return all probe questions grouped by category/specialty."""
        questions = []
        for category, specialties in self._probes.items():
            if isinstance(specialties, dict):
                for specialty, qs in specialties.items():
                    if isinstance(qs, list):
                        for q in qs:
                            questions.append({
                                "category": category,
                                "specialty": specialty,
                                "question": q.get("q", ""),
                                "expected_contains": q.get("expected_contains", []),
                                "type": q.get("type", "text"),
                            })
            elif isinstance(specialties, list):
                # e.g. vision probes are a flat list
                for q in specialties:
                    questions.append({
                        "category": category,
                        "specialty": category,
                        "question": q.get("q", ""),
                        "expected_contains": q.get("expected_contains", []),
                        "type": q.get("type", "text"),
                    })
        return questions

    def score_response(
        self, response: str, expected_keywords: list[str]
    ) -> tuple[float, list[str]]:
        """Score a probe response using keyword matching.

        Returns (accuracy, matched_keywords).
        """
        if not expected_keywords:
            return 1.0, []

        response_lower = response.lower()
        matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
        accuracy = len(matched) / len(expected_keywords)
        return accuracy, matched

    async def probe_model(
        self,
        model_name: str,
        query_fn: Any,  # async callable(question: str) -> str
    ) -> ModelProbeReport:
        """Probe a model with all questions and return capability report.

        The query_fn should be an async callable that sends a question to the
        model and returns the text response. The actual model call is handled
        externally so probing stays provider-agnostic.

        Results are stored but NOT auto-activated -- human approval via
        dashboard is required before the model can serve these specialties.
        """
        report = ModelProbeReport(model_name=model_name)
        specialty_scores: dict[str, list[float]] = {}

        questions = [q for q in self.get_probe_questions() if q.get("type") != "vision_test"]

        # Run all probe queries in PARALLEL for speed (~5s vs ~40s sequential)
        async def run_one(q):
            try:
                response = await query_fn(q["question"])
            except Exception as exc:
                logger.error("Probe failed for %s on %r: %s", model_name, q["question"], exc)
                response = ""
            expected = q.get("expected_contains", [])
            accuracy, matched = self.score_response(response, expected)
            return ProbeResult(
                specialty=q["specialty"],
                question=q["question"],
                response=response,
                expected_keywords=expected,
                matched_keywords=matched,
                accuracy=accuracy,
            )

        import asyncio
        results = await asyncio.gather(*[run_one(q) for q in questions])

        for result in results:
            report.results.append(result)
            spec = result.specialty
            if spec not in specialty_scores:
                specialty_scores[spec] = []
            specialty_scores[spec].append(result.accuracy)

        # Average accuracy per specialty
        for spec, scores in specialty_scores.items():
            report.specialty_scores[spec] = sum(scores) / len(scores) if scores else 0.0

        self._reports[model_name] = report
        logger.info(
            "Probed %s: %s",
            model_name,
            {k: f"{v:.0%}" for k, v in report.specialty_scores.items()},
        )
        return report

    def get_report(self, model_name: str) -> Optional[ModelProbeReport]:
        return self._reports.get(model_name)

    def list_reports(self) -> list[ModelProbeReport]:
        return list(self._reports.values())

    def approve_model(self, model_name: str) -> bool:
        """Human approval gate: mark model capabilities as approved."""
        report = self._reports.get(model_name)
        if report is None:
            return False
        report.approved = True
        return True


# Singleton
capability_prober = CapabilityProber()
