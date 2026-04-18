#!/usr/bin/env python3
"""Quick benchmark on real dataset samples — verify model selection correctness.

Tests 60 samples: 20 PathVQA + 20 VQA-RAD + 20 PubMedQA.
Checks if router picks correct specialty and appropriate model.

Run on A100: python -m benchmarks.real_dataset_eval
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.router import MedVisionRouter
from src.registry import model_registry
from src.prompts import prompt_manager
from src.explainability import explain_decision

CONFIG = ROOT / "config"
DATASET_DIR = Path.home() / "med-datasets"


async def run():
    model_registry.seed_from_config(CONFIG / "models.yaml")
    prompt_manager.load(CONFIG / "prompt_templates.yaml")
    prompt_manager.load_taxonomy(CONFIG / "taxonomy.yaml")

    router = MedVisionRouter()
    await router.initialize()

    results = []

    # PathVQA
    try:
        from datasets import load_from_disk
        pvqa = load_from_disk(str(DATASET_DIR / "pathvqa-5k"))
        print("=== PathVQA (expect: pathology / medical, vision model) ===")
        for i in range(min(20, len(pvqa))):
            q = pvqa[i]["question"]
            d = await router.route(messages=[{"role": "user", "content": q}])
            ok = "pathol" in d.specialty.lower() or d.specialty.startswith("medical.")
            model_ok = "vision" in (model_registry.get_model(d.model_name).capabilities if model_registry.get_model(d.model_name) else [])
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} spec={d.specialty:30s} model={d.model_name:15s} q={q[:55]}")
            results.append({"ds": "pathvqa", "specialty_ok": ok, "model_has_vision": model_ok,
                           "specialty": d.specialty, "model": d.model_name, "query": q[:60]})
    except Exception as e:
        print(f"PathVQA skip: {e}")

    # VQA-RAD
    try:
        vqarad = load_from_disk(str(DATASET_DIR / "vqa-rad"))
        print("\n=== VQA-RAD (expect: radiology / medical, vision model) ===")
        for i in range(min(20, len(vqarad))):
            q = str(vqarad[i]["question"])
            d = await router.route(messages=[{"role": "user", "content": q}])
            ok = "radiol" in d.specialty.lower() or d.specialty.startswith("medical.")
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} spec={d.specialty:30s} model={d.model_name:15s} q={q[:55]}")
            results.append({"ds": "vqarad", "specialty_ok": ok, "specialty": d.specialty,
                           "model": d.model_name, "query": q[:60]})
    except Exception as e:
        print(f"VQA-RAD skip: {e}")

    # PubMedQA
    try:
        pubmed = load_from_disk(str(DATASET_DIR / "pubmedqa-labeled"))
        print("\n=== PubMedQA (expect: medical specialty) ===")
        for i in range(min(20, len(pubmed))):
            q = str(pubmed[i]["question"])
            d = await router.route(messages=[{"role": "user", "content": q}])
            ok = d.specialty.startswith("medical.")
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} spec={d.specialty:30s} model={d.model_name:15s} q={q[:55]}")
            results.append({"ds": "pubmedqa", "specialty_ok": ok, "specialty": d.specialty,
                           "model": d.model_name, "query": q[:60]})
    except Exception as e:
        print(f"PubMedQA skip: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  REAL DATASET BENCHMARK RESULTS")
    print("=" * 70)
    for ds in ["pathvqa", "vqarad", "pubmedqa"]:
        subset = [r for r in results if r["ds"] == ds]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["specialty_ok"])
        total = len(subset)
        pct = correct / total * 100
        print(f"  {ds:15s}: {correct}/{total} = {pct:.0f}%")

        # Show model distribution
        models = {}
        for r in subset:
            models[r["model"]] = models.get(r["model"], 0) + 1
        for m, c in sorted(models.items(), key=lambda x: -x[1]):
            print(f"    {m:20s}: {c}")

        # Show specialty distribution
        specs = {}
        for r in subset:
            specs[r["specialty"]] = specs.get(r["specialty"], 0) + 1
        for s, c in sorted(specs.items(), key=lambda x: -x[1]):
            print(f"    → {s:25s}: {c}")

    overall = sum(1 for r in results if r["specialty_ok"])
    print(f"\n  Overall: {overall}/{len(results)} = {overall/len(results)*100:.0f}%")

    # Explainability sample
    if results:
        print("\n--- Sample Explanation ---")
        sample_q = results[0]["query"]
        d = await router.route(messages=[{"role": "user", "content": sample_q}])
        exp = explain_decision(d)
        print(exp.text)

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run())
