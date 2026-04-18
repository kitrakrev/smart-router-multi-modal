#!/usr/bin/env python3
"""Benchmark runner for PathVQA, PubMedQA, VQA-RAD.

For each benchmark:
  1. Load dataset (partial -- max 5K samples)
  2. For each query: run router -> get decision (specialty, model, prompt)
  3. If running end-to-end: send to model, compare answer to ground truth
  4. If routing-only: check if specialty/model choice is reasonable

Metrics:
  - Routing accuracy: did router pick correct specialty?
  - Model accuracy: did picked model + prompt get answer right?
  - Cost per 1K queries
  - Avg routing latency
  - Per-specialty breakdown

CLI:
  python -m optimize.evaluate --benchmark pathvqa --mode routing-only
  python -m optimize.evaluate --benchmark pubmedqa --mode end-to-end --api-base http://localhost:8000/v1
  python -m optimize.evaluate --all
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.signals import run_all_signals
from src.router import Router

# ---------------------------------------------------------------------------
# Dataset loading (partial -- max 5K)
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "optimize" / "results"


def load_dataset_partial(name: str, max_samples: int = 5000) -> list[dict]:
    """Load a partial dataset for evaluation.

    Returns list of dicts with keys: question, answer, context, expected_specialty, has_image.
    """
    from datasets import load_dataset

    samples = []

    if name == "pathvqa":
        ds = load_dataset("flaviagiammarino/path-vqa", split="train")
        ds = ds.select(range(min(len(ds), max_samples)))
        for row in ds:
            samples.append({
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "context": "",
                "expected_specialty": "pathology",
                "expected_domain": "medical",
                "has_image": True,  # PathVQA is a vision dataset
                "dataset": "pathvqa",
            })

    elif name == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        ds = ds.select(range(min(len(ds), max_samples)))
        for row in ds:
            ctx = row.get("context", {})
            context_text = ""
            if isinstance(ctx, dict):
                contexts = ctx.get("contexts", [])
                context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
            elif isinstance(ctx, list):
                context_text = " ".join(str(c) for c in ctx)

            samples.append({
                "question": str(row.get("question", "")),
                "answer": str(row.get("final_decision", "")),
                "context": context_text[:500],
                "expected_specialty": "general_medicine",
                "expected_domain": "medical",
                "has_image": False,
                "dataset": "pubmedqa",
            })

    elif name == "vqa_rad":
        # VQA-RAD is a smaller radiology VQA dataset
        try:
            ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
            ds = ds.select(range(min(len(ds), max_samples)))
            for row in ds:
                samples.append({
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "context": "",
                    "expected_specialty": "radiology",
                    "expected_domain": "medical",
                    "has_image": True,
                    "dataset": "vqa_rad",
                })
        except Exception as e:
            print(f"  [WARN] Could not load VQA-RAD: {e}")

    else:
        raise ValueError(f"Unknown benchmark: {name}. Available: pathvqa, pubmedqa, vqa_rad")

    print(f"  Loaded {len(samples)} samples from {name}")
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    benchmark: str = ""
    mode: str = ""
    total_samples: int = 0
    routing_correct: int = 0
    domain_correct: int = 0
    vision_detected: int = 0
    vision_expected: int = 0
    model_correct: int = 0
    model_evaluated: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    specialty_breakdown: dict[str, dict] = field(default_factory=dict)
    errors: int = 0
    timestamp: str = ""

    @property
    def routing_accuracy(self) -> float:
        return self.routing_correct / self.total_samples if self.total_samples else 0.0

    @property
    def domain_accuracy(self) -> float:
        return self.domain_correct / self.total_samples if self.total_samples else 0.0

    @property
    def vision_detection_rate(self) -> float:
        return self.vision_detected / self.vision_expected if self.vision_expected else 0.0

    @property
    def model_accuracy(self) -> float:
        return self.model_correct / self.model_evaluated if self.model_evaluated else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def cost_per_1k(self) -> float:
        return (self.total_cost_usd / self.total_samples * 1000) if self.total_samples else 0.0

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "mode": self.mode,
            "total_samples": self.total_samples,
            "routing_accuracy": round(self.routing_accuracy, 4),
            "domain_accuracy": round(self.domain_accuracy, 4),
            "vision_detection_rate": round(self.vision_detection_rate, 4),
            "model_accuracy": round(self.model_accuracy, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cost_per_1k_queries_usd": round(self.cost_per_1k, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "errors": self.errors,
            "specialty_breakdown": self.specialty_breakdown,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Medical specialty mapping for evaluation
# ---------------------------------------------------------------------------

# Map expected specialties to acceptable router decision names
SPECIALTY_ACCEPTABLE_DECISIONS: dict[str, set[str]] = {
    "pathology": {
        "pathology", "medical_vision", "vision_task",
        "medical_specialist", "medical_text",
    },
    "radiology": {
        "radiology", "medical_vision", "vision_task",
        "medical_specialist", "medical_text",
    },
    "general_medicine": {
        "general_medicine", "medical_text", "medical_specialist",
        "clinical_qa", "complex_reasoning",
    },
    "cardiology": {
        "cardiology", "medical_specialist", "medical_text",
        "clinical_qa",
    },
    "dermatology": {
        "dermatology", "medical_vision", "vision_task",
        "medical_specialist",
    },
    "ophthalmology": {
        "ophthalmology", "medical_vision", "vision_task",
        "medical_specialist",
    },
    "emergency": {
        "emergency", "medical_specialist", "medical_text",
    },
    "pharmacology": {
        "pharmacology", "medical_specialist", "medical_text",
        "tool_usage",
    },
}

# Acceptable model capabilities per expected specialty
SPECIALTY_REQUIRED_CAPS: dict[str, set[str]] = {
    "pathology": {"medical", "vision"},
    "radiology": {"medical", "vision"},
    "general_medicine": {"medical", "text"},
    "cardiology": {"medical", "text"},
    "dermatology": {"medical", "vision"},
    "ophthalmology": {"medical", "vision"},
    "emergency": {"medical", "text"},
    "pharmacology": {"medical", "text"},
}


# ---------------------------------------------------------------------------
# Router evaluation (routing-only mode)
# ---------------------------------------------------------------------------


def _build_messages(sample: dict) -> list[dict]:
    """Convert a benchmark sample into OpenAI-style messages."""
    content_parts: list[dict | str] = []
    question = sample["question"]
    context = sample.get("context", "")

    if context:
        text = f"{context}\n\nQuestion: {question}"
    else:
        text = question

    if sample.get("has_image"):
        # Simulate an image in the message (we don't have actual images for routing test)
        content_parts.append({"type": "text", "text": text})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
        })
        return [{"role": "user", "content": content_parts}]
    else:
        return [{"role": "user", "content": text}]


async def evaluate_routing(
    router: Router,
    samples: list[dict],
    verbose: bool = False,
) -> BenchmarkMetrics:
    """Evaluate routing decisions (no model inference).

    Checks:
      - Is the domain correctly identified as medical?
      - Is the specialty decision reasonable?
      - Are vision queries routed to vision-capable models?
    """
    metrics = BenchmarkMetrics()
    metrics.mode = "routing-only"

    specialty_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()

        messages = _build_messages(sample)
        query_text = sample["question"]

        try:
            # Run signals
            signals = await run_all_signals(messages)

            # Run router
            decision = router.decide(signals, query_text=query_text)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.latencies_ms.append(elapsed_ms)
            metrics.total_samples += 1
            metrics.total_cost_usd += decision.estimated_cost

            expected_specialty = sample["expected_specialty"]
            expected_domain = sample.get("expected_domain", "medical")

            # Check domain detection
            domain_sig = next((s for s in signals if s.name == "domain"), None)
            if domain_sig:
                detected_domain = domain_sig.metadata.get("domain", "")
                if detected_domain == expected_domain or detected_domain == "medical":
                    metrics.domain_correct += 1

            # Check specialty routing
            decision_name = decision.decision_name
            acceptable = SPECIALTY_ACCEPTABLE_DECISIONS.get(expected_specialty, set())
            # Also accept if the decision name contains the specialty
            is_correct = (
                decision_name in acceptable
                or expected_specialty in decision_name
                or decision_name == expected_specialty
            )
            if is_correct:
                metrics.routing_correct += 1
            specialty_counts[expected_specialty]["total"] += 1
            if is_correct:
                specialty_counts[expected_specialty]["correct"] += 1

            # Check vision detection
            if sample.get("has_image"):
                metrics.vision_expected += 1
                vision_sig = next((s for s in signals if s.name == "vision"), None)
                if vision_sig and vision_sig.metadata.get("detected"):
                    metrics.vision_detected += 1

            if verbose and (i < 10 or not is_correct):
                mark = "OK" if is_correct else "MISS"
                print(
                    f"  [{mark}] Q: {query_text[:60]}... "
                    f"-> {decision_name} (expected: {expected_specialty}) "
                    f"model={decision.selected_model}"
                )

        except Exception as e:
            metrics.errors += 1
            if verbose:
                print(f"  [ERR] {query_text[:40]}...: {e}")

        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(samples)}...")

    # Build specialty breakdown
    for spec, counts in specialty_counts.items():
        total = counts["total"]
        correct = counts["correct"]
        metrics.specialty_breakdown[spec] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }

    return metrics


# ---------------------------------------------------------------------------
# End-to-end evaluation (with model inference)
# ---------------------------------------------------------------------------


async def evaluate_end_to_end(
    router: Router,
    samples: list[dict],
    api_base: str = "http://localhost:8000/v1",
    verbose: bool = False,
) -> BenchmarkMetrics:
    """End-to-end evaluation: route + infer + score.

    Requires a running model server at api_base.
    """
    import httpx

    metrics = await evaluate_routing(router, samples, verbose=verbose)
    metrics.mode = "end-to-end"

    # Now run inference for a subset (cap at 200 for speed)
    eval_subset = samples[:200]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, sample in enumerate(eval_subset):
            messages = _build_messages(sample)
            query_text = sample["question"]

            try:
                signals = await run_all_signals(messages)
                decision = router.decide(signals, query_text=query_text)

                if decision.blocked:
                    continue

                # Send to model
                payload = {
                    "model": decision.selected_model,
                    "messages": [{"role": "user", "content": query_text}],
                    "temperature": decision.inference_config.get("temperature", 0.1),
                    "max_tokens": decision.inference_config.get("max_tokens", 256),
                }

                response = await client.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    model_answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # Score
                    from optimize.dspy_optimizer import score_medical

                    score = score_medical(model_answer, sample["answer"])
                    metrics.model_evaluated += 1
                    if score >= 0.5:
                        metrics.model_correct += 1

                    if verbose and i < 10:
                        print(
                            f"  [E2E] Q: {query_text[:50]}... "
                            f"Model: {decision.selected_model} "
                            f"Score: {score:.2f}"
                        )

            except Exception as e:
                if verbose:
                    print(f"  [E2E-ERR] {query_text[:40]}...: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------


async def run_benchmark(
    benchmark_name: str,
    mode: str = "routing-only",
    config_path: str = "config/config.yaml",
    max_samples: int = 5000,
    api_base: str = "http://localhost:8000/v1",
    verbose: bool = False,
) -> dict:
    """Run a full benchmark and return results."""
    print(f"\n{'='*60}")
    print(f"  Benchmark: {benchmark_name} (mode: {mode})")
    print(f"{'='*60}")

    # Load dataset
    samples = load_dataset_partial(benchmark_name, max_samples=max_samples)

    # Create router
    full_config_path = str(ROOT / config_path)
    router = Router(config_path=full_config_path)

    # Run evaluation
    if mode == "end-to-end":
        metrics = await evaluate_end_to_end(
            router, samples, api_base=api_base, verbose=verbose
        )
    else:
        metrics = await evaluate_routing(router, samples, verbose=verbose)

    metrics.benchmark = benchmark_name
    metrics.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Print report
    result = metrics.to_dict()
    print(f"\n  Results for {benchmark_name}:")
    print(f"    Samples:           {result['total_samples']}")
    print(f"    Routing accuracy:  {result['routing_accuracy']:.2%}")
    print(f"    Domain accuracy:   {result['domain_accuracy']:.2%}")
    print(f"    Vision detection:  {result['vision_detection_rate']:.2%}")
    if result["model_accuracy"] > 0:
        print(f"    Model accuracy:    {result['model_accuracy']:.2%}")
    print(f"    Avg latency:       {result['avg_latency_ms']:.1f} ms")
    print(f"    Cost per 1K:       ${result['cost_per_1k_queries_usd']:.4f}")
    print(f"    Errors:            {result['errors']}")

    if result["specialty_breakdown"]:
        print(f"\n    Per-specialty breakdown:")
        for spec, info in sorted(result["specialty_breakdown"].items()):
            print(f"      {spec:20s}: {info['correct']}/{info['total']} ({info['accuracy']:.2%})")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"eval_{benchmark_name}_{mode}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to: {result_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for MedVisionRouter"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["pathvqa", "pubmedqa", "vqa_rad"],
        help="Benchmark dataset to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="routing-only",
        choices=["routing-only", "end-to-end"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Router config path (relative to project root)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Max samples to evaluate",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for end-to-end mode",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample results",
    )

    args = parser.parse_args()

    if args.all:
        all_results = {}
        for bm in ["pathvqa", "pubmedqa"]:
            result = asyncio.run(
                run_benchmark(
                    bm,
                    mode=args.mode,
                    config_path=args.config,
                    max_samples=args.max_samples,
                    api_base=args.api_base,
                    verbose=args.verbose,
                )
            )
            all_results[bm] = result

        # Save combined results
        combined_path = RESULTS_DIR / "eval_all.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined results saved to: {combined_path}")
        return

    if not args.benchmark:
        parser.error("Provide --benchmark or use --all")
        return

    asyncio.run(
        run_benchmark(
            args.benchmark,
            mode=args.mode,
            config_path=args.config,
            max_samples=args.max_samples,
            api_base=args.api_base,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
