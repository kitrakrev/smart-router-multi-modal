#!/usr/bin/env python3
"""PubMedQA routing benchmark -- 1K labeled samples.

Evaluates the MedVisionRouter on clinical text QA:
  1. Did router pick medical specialty (not general)?
  2. Did router pick medical-capable model?
  3. Is prompt appropriate for clinical QA?

Report: medical detection rate, specialty accuracy.

CLI:
  python -m benchmarks.pubmedqa
  python -m benchmarks.pubmedqa --max-samples 500 --verbose
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


def load_pubmedqa(max_samples: int = 1000) -> list[dict]:
    """Load partial PubMedQA labeled dataset."""
    from datasets import load_dataset

    print("Loading PubMedQA (pqa_labeled) from HuggingFace...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    total = len(ds)
    n = min(total, max_samples)
    ds = ds.select(range(n))
    print(f"  Selected {n}/{total} samples")

    samples = []
    for row in ds:
        q = str(row.get("question", ""))
        answer = str(row.get("final_decision", ""))
        long_answer = str(row.get("long_answer", ""))

        # Extract context
        ctx = row.get("context", {})
        context_text = ""
        if isinstance(ctx, dict):
            contexts = ctx.get("contexts", [])
            context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        elif isinstance(ctx, list):
            context_text = " ".join(str(c) for c in ctx)

        if q.strip() and answer.strip():
            samples.append({
                "question": q,
                "answer": answer,
                "long_answer": long_answer[:300],
                "context": context_text[:500],
                "has_image": False,
                "expected_domain": "medical",
            })

    print(f"  Valid samples: {len(samples)}")
    return samples


async def run_pubmedqa_benchmark(
    config_path: str = "config/config.yaml",
    max_samples: int = 1000,
    verbose: bool = False,
) -> dict:
    """Run PubMedQA routing benchmark."""
    samples = load_pubmedqa(max_samples)

    # Initialize v2 router
    CONFIG_DIR = ROOT / "config"
    model_registry.seed_from_config(CONFIG_DIR / "models.yaml")
    prompt_manager.load(CONFIG_DIR / "prompt_templates.yaml")
    prompt_manager.load_taxonomy(CONFIG_DIR / "taxonomy.yaml")
    router = MedVisionRouter()
    await router.initialize()

    metrics = {
        "total": 0,
        "medical_detected": 0,
        "medical_model_selected": 0,
        "reasoning_model_selected": 0,
        "domain_scores": defaultdict(int),
        "decision_distribution": defaultdict(int),
        "model_distribution": defaultdict(int),
        "latencies_ms": [],
        "errors": 0,
    }

    # Identify medical-capable and reasoning-capable models
    import yaml

    models_yaml = ROOT / "config" / "models.yaml"
    medical_models = set()
    reasoning_models = set()
    if models_yaml.exists():
        with open(models_yaml) as f:
            models_cfg = yaml.safe_load(f) or {}
        for m in models_cfg.get("models", []):
            caps = m.get("capabilities", [])
            if "medical" in caps:
                medical_models.add(m["name"])
            if "reasoning" in caps:
                reasoning_models.add(m["name"])

    print(f"\nRunning PubMedQA benchmark on {len(samples)} samples...")
    print(f"  Medical models: {medical_models or '(none configured)'}")
    print(f"  Reasoning models: {reasoning_models or '(none configured)'}")

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()

        # Build message with context
        text = sample["question"]
        if sample.get("context"):
            text = f"Context: {sample['context'][:300]}\n\nQuestion: {text}"

        messages = [{"role": "user", "content": text}]

        try:
            decision = await router.route(messages=messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            metrics["total"] += 1
            metrics["latencies_ms"].append(elapsed_ms)
            metrics["decision_distribution"][decision.specialty] += 1
            metrics["model_distribution"][decision.model_name] += 1

            # 1. Medical detection: did text signal find medical specialty?
            is_medical = decision.specialty.startswith("medical.")
            if is_medical:
                metrics["medical_detected"] += 1
            domain = "medical" if is_medical else "general"
            metrics["domain_scores"][domain] += 1

            # 2. Medical model selection
            if decision.model_name in medical_models:
                metrics["medical_model_selected"] += 1

            # 3. Reasoning model selection
            if decision.model_name in reasoning_models:
                metrics["reasoning_model_selected"] += 1

            if verbose and i < 10:
                print(
                    f"  Q: {sample['question'][:55]}... "
                    f"-> specialty={decision.specialty} "
                    f"model={decision.model_name}"
                )

        except Exception as e:
            metrics["errors"] += 1
            if verbose:
                print(f"  [ERR] {sample['question'][:40]}...: {e}")

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(samples)}")

    # Compute report
    total = metrics["total"]
    avg_lat = sum(metrics["latencies_ms"]) / len(metrics["latencies_ms"]) if metrics["latencies_ms"] else 0

    report = {
        "benchmark": "PubMedQA",
        "total_samples": total,
        "medical_detection_rate": round(metrics["medical_detected"] / total, 4) if total else 0,
        "medical_model_rate": round(metrics["medical_model_selected"] / total, 4) if total else 0,
        "reasoning_model_rate": round(metrics["reasoning_model_selected"] / total, 4) if total else 0,
        "avg_routing_latency_ms": round(avg_lat, 2),
        "p95_routing_latency_ms": round(
            sorted(metrics["latencies_ms"])[int(len(metrics["latencies_ms"]) * 0.95)]
            if metrics["latencies_ms"]
            else 0,
            2,
        ),
        "errors": metrics["errors"],
        "domain_distribution": dict(metrics["domain_scores"]),
        "decision_distribution": dict(metrics["decision_distribution"]),
        "model_distribution": dict(metrics["model_distribution"]),
    }

    # Print report
    print(f"\n{'='*60}")
    print(f"  PubMedQA Benchmark Results")
    print(f"{'='*60}")
    print(f"  Samples:               {report['total_samples']}")
    print(f"  Medical detection:      {report['medical_detection_rate']:.2%}")
    print(f"  Medical model rate:     {report['medical_model_rate']:.2%}")
    print(f"  Reasoning model rate:   {report['reasoning_model_rate']:.2%}")
    print(f"  Avg latency:            {report['avg_routing_latency_ms']:.1f} ms")
    print(f"  P95 latency:            {report['p95_routing_latency_ms']:.1f} ms")
    print(f"  Errors:                 {report['errors']}")
    print(f"\n  Domain signal distribution:")
    for dom, cnt in sorted(report["domain_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {dom:20s}: {cnt:5d} ({cnt/total:.1%})")
    print(f"\n  Decision distribution:")
    for dec, cnt in sorted(report["decision_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {dec:25s}: {cnt:5d} ({cnt/total:.1%})")
    print(f"\n  Model distribution:")
    for model, cnt in sorted(report["model_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {model:25s}: {cnt:5d} ({cnt/total:.1%})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "pubmedqa_benchmark.json"
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved to: {result_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="PubMedQA routing benchmark")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_pubmedqa_benchmark(
            config_path=args.config,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
