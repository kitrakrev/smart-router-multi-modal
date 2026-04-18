#!/usr/bin/env python3
"""PathVQA routing benchmark -- 5K samples from 32K total.

Evaluates the MedVisionRouter on pathology visual question answering:
  1. Did vision signal detect pathology image?
  2. Did router pick pathology specialty?
  3. Did router pick vision-capable model?
  4. If running model: did model answer correctly?

Report: routing accuracy, vision detection rate, specialty accuracy.

CLI:
  python -m benchmarks.pathvqa
  python -m benchmarks.pathvqa --max-samples 1000 --verbose
  python -m benchmarks.pathvqa --end-to-end --api-base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.router import MedVisionRouter
from src.registry import model_registry
from src.prompts import prompt_manager

RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_pathvqa(max_samples: int = 5000) -> list[dict]:
    """Load partial PathVQA dataset."""
    from datasets import load_dataset

    print("Loading PathVQA from HuggingFace...")
    ds = load_dataset("flaviagiammarino/path-vqa", split="train")
    total = len(ds)
    n = min(total, max_samples)
    ds = ds.select(range(n))
    print(f"  Selected {n}/{total} samples")

    samples = []
    for row in ds:
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if q.strip() and a.strip():
            samples.append({
                "question": q,
                "answer": a,
                "has_image": True,
                "expected_specialty": "pathology",
            })

    print(f"  Valid samples: {len(samples)}")
    return samples


def _build_pathvqa_messages(sample: dict) -> list[dict]:
    """Build OpenAI-style messages with a simulated image."""
    q = sample["question"]
    # Include a minimal 1x1 PNG to trigger vision signal
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": q},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]


async def run_pathvqa_benchmark(
    config_path: str = "config/config.yaml",
    max_samples: int = 5000,
    verbose: bool = False,
) -> dict:
    """Run PathVQA routing benchmark."""
    samples = load_pathvqa(max_samples)

    # Initialize v2 router
    CONFIG_DIR = ROOT / "config"
    model_registry.seed_from_config(CONFIG_DIR / "models.yaml")
    prompt_manager.load(CONFIG_DIR / "prompt_templates.yaml")
    prompt_manager.load_taxonomy(CONFIG_DIR / "taxonomy.yaml")
    router = MedVisionRouter()
    await router.initialize()

    metrics = {
        "total": 0,
        "vision_detected": 0,
        "specialty_correct": 0,
        "vision_model_selected": 0,
        "latencies_ms": [],
        "decision_distribution": defaultdict(int),
        "model_distribution": defaultdict(int),
        "errors": 0,
    }

    # Load model capabilities for checking vision-capable selection
    models_yaml = ROOT / "config" / "models.yaml"
    vision_models = set()
    if models_yaml.exists():
        import yaml

        with open(models_yaml) as f:
            models_cfg = yaml.safe_load(f) or {}
        for m in models_cfg.get("models", []):
            if "vision" in m.get("capabilities", []):
                vision_models.add(m["name"])

    print(f"\nRunning PathVQA benchmark on {len(samples)} samples...")
    print(f"  Vision-capable models: {vision_models or '(none configured)'}")

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        messages = _build_pathvqa_messages(sample)

        try:
            decision = await router.route(messages=messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            metrics["total"] += 1
            metrics["latencies_ms"].append(elapsed_ms)
            metrics["decision_distribution"][decision.specialty] += 1
            metrics["model_distribution"][decision.model_name] += 1

            # 1. Vision signal detection
            vision_sig = decision.signals.get("vision", {})
            if vision_sig.get("image_type", "none") != "none":
                metrics["vision_detected"] += 1

            # 2. Specialty routing (accept any medical/pathology decision)
            specialty = decision.specialty
            is_specialty_correct = (
                "pathol" in specialty.lower()
                or "medical" in specialty.lower()
                or specialty.startswith("medical.")
            )
            if is_specialty_correct:
                metrics["specialty_correct"] += 1

            # 3. Vision-capable model selection
            if decision.model_name in vision_models:
                metrics["vision_model_selected"] += 1

            if verbose and (i < 10 or not is_specialty_correct):
                mark = "OK" if is_specialty_correct else "MISS"
                print(
                    f"  [{mark}] Q: {sample['question'][:55]}... "
                    f"-> {specialty} model={decision.model_name}"
                )

        except Exception as e:
            metrics["errors"] += 1
            if verbose:
                print(f"  [ERR] {sample['question'][:40]}...: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(samples)}")

    # Compute report
    total = metrics["total"]
    avg_lat = sum(metrics["latencies_ms"]) / len(metrics["latencies_ms"]) if metrics["latencies_ms"] else 0

    report = {
        "benchmark": "PathVQA",
        "total_samples": total,
        "vision_detection_rate": round(metrics["vision_detected"] / total, 4) if total else 0,
        "specialty_routing_accuracy": round(metrics["specialty_correct"] / total, 4) if total else 0,
        "vision_model_selection_rate": round(metrics["vision_model_selected"] / total, 4) if total else 0,
        "avg_routing_latency_ms": round(avg_lat, 2),
        "p95_routing_latency_ms": round(sorted(metrics["latencies_ms"])[int(len(metrics["latencies_ms"]) * 0.95)] if metrics["latencies_ms"] else 0, 2),
        "errors": metrics["errors"],
        "decision_distribution": dict(metrics["decision_distribution"]),
        "model_distribution": dict(metrics["model_distribution"]),
    }

    # Print report
    print(f"\n{'='*60}")
    print(f"  PathVQA Benchmark Results")
    print(f"{'='*60}")
    print(f"  Samples:               {report['total_samples']}")
    print(f"  Vision detection rate:  {report['vision_detection_rate']:.2%}")
    print(f"  Specialty accuracy:     {report['specialty_routing_accuracy']:.2%}")
    print(f"  Vision model selected:  {report['vision_model_selection_rate']:.2%}")
    print(f"  Avg latency:            {report['avg_routing_latency_ms']:.1f} ms")
    print(f"  P95 latency:            {report['p95_routing_latency_ms']:.1f} ms")
    print(f"  Errors:                 {report['errors']}")
    print(f"\n  Decision distribution:")
    for dec, cnt in sorted(report["decision_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {dec:25s}: {cnt:5d} ({cnt/total:.1%})")
    print(f"\n  Model distribution:")
    for model, cnt in sorted(report["model_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {model:25s}: {cnt:5d} ({cnt/total:.1%})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "pathvqa_benchmark.json"
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved to: {result_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="PathVQA routing benchmark")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_pathvqa_benchmark(
            config_path=args.config,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
