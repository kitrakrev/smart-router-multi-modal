#!/usr/bin/env python3
"""Real image benchmark — encode actual dataset images as base64 and route."""
import asyncio, sys, base64, io, time, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.router import MedVisionRouter
from src.registry import model_registry
from src.prompts import prompt_manager
from src.explainability import explain_decision

CONFIG = ROOT / "config"
DATASET_DIR = Path.home() / "med-datasets"

def pil_to_base64(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

async def run():
    model_registry.seed_from_config(CONFIG / "models.yaml")
    prompt_manager.load(CONFIG / "prompt_templates.yaml")
    prompt_manager.load_taxonomy(CONFIG / "taxonomy.yaml")
    router = MedVisionRouter()
    await router.initialize()

    from datasets import load_from_disk
    results = []

    # PathVQA with real images
    pvqa = load_from_disk(str(DATASET_DIR / "pathvqa-5k"))
    print("=== PathVQA WITH REAL IMAGES (100 samples) ===")
    t0 = time.time()
    for i in range(100):
        q = pvqa[i]["question"]
        img = pvqa[i]["image"]
        img_b64 = pil_to_base64(img)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        ]}]
        d = await router.route(messages=messages)
        ok = d.specialty.startswith("medical.")
        vis = d.signals.get("vision", {})
        img_type = vis.get("image_type", "none")
        results.append({"ds": "pathvqa", "ok": ok, "spec": d.specialty,
                        "model": d.model_name, "img_type": img_type, "q": q[:50]})
        if i < 10 or not ok:
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} spec={d.specialty:28s} img={img_type:12s} model={d.model_name:15s} q={q[:45]}")
    elapsed_pvqa = time.time() - t0
    pvqa_ok = sum(1 for r in results if r["ds"] == "pathvqa" and r["ok"])
    print(f"  PathVQA: {pvqa_ok}/100 = {pvqa_ok}% in {elapsed_pvqa:.1f}s")

    # VQA-RAD with real images
    vqarad = load_from_disk(str(DATASET_DIR / "vqa-rad"))
    print("\n=== VQA-RAD WITH REAL IMAGES (100 samples) ===")
    t0 = time.time()
    for i in range(100):
        q = str(vqarad[i]["question"])
        img = vqarad[i]["image"]
        img_b64 = pil_to_base64(img)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        ]}]
        d = await router.route(messages=messages)
        ok = d.specialty.startswith("medical.")
        vis = d.signals.get("vision", {})
        img_type = vis.get("image_type", "none")
        results.append({"ds": "vqarad", "ok": ok, "spec": d.specialty,
                        "model": d.model_name, "img_type": img_type, "q": q[:50]})
        if i < 10 or not ok:
            mark = "OK" if ok else "MISS"
            print(f"  {mark:4s} spec={d.specialty:28s} img={img_type:12s} model={d.model_name:15s} q={q[:45]}")
    elapsed_vqarad = time.time() - t0
    vqarad_ok = sum(1 for r in results if r["ds"] == "vqarad" and r["ok"])
    print(f"  VQA-RAD: {vqarad_ok}/100 = {vqarad_ok}% in {elapsed_vqarad:.1f}s")

    # PubMedQA text only (200 samples for volume)
    pubmed = load_from_disk(str(DATASET_DIR / "pubmedqa-labeled"))
    print("\n=== PubMedQA TEXT (200 samples) ===")
    t0 = time.time()
    for i in range(200):
        q = str(pubmed[i]["question"])
        d = await router.route(messages=[{"role": "user", "content": q}])
        ok = d.specialty.startswith("medical.")
        results.append({"ds": "pubmedqa", "ok": ok, "spec": d.specialty,
                        "model": d.model_name, "img_type": "", "q": q[:50]})
        if not ok:
            print(f"  MISS spec={d.specialty:28s} q={q[:55]}")
    elapsed_pubmed = time.time() - t0
    pubmed_ok = sum(1 for r in results if r["ds"] == "pubmedqa" and r["ok"])
    print(f"  PubMedQA: {pubmed_ok}/200 = {pubmed_ok/2:.0f}% in {elapsed_pubmed:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  REAL IMAGE BENCHMARK RESULTS (400 samples)")
    print("=" * 70)
    for ds in ["pathvqa", "vqarad", "pubmedqa"]:
        subset = [r for r in results if r["ds"] == ds]
        correct = sum(1 for r in subset if r["ok"])
        total = len(subset)
        print(f"\n  {ds} ({total} samples): {correct}/{total} = {correct/total*100:.1f}%")

        # Model distribution
        models = defaultdict(int)
        for r in subset:
            models[r["model"]] += 1
        for m, c in sorted(models.items(), key=lambda x: -x[1]):
            print(f"    model {m:20s}: {c:4d} ({c/total*100:.0f}%)")

        # Image type distribution (for vision datasets)
        if ds in ("pathvqa", "vqarad"):
            img_types = defaultdict(int)
            for r in subset:
                img_types[r["img_type"]] += 1
            print(f"    Image types detected:")
            for t, c in sorted(img_types.items(), key=lambda x: -x[1]):
                print(f"      {t:15s}: {c:4d}")

        # Specialty distribution
        specs = defaultdict(int)
        for r in subset:
            specs[r["spec"]] += 1
        print(f"    Specialties routed:")
        for s, c in sorted(specs.items(), key=lambda x: -x[1])[:5]:
            print(f"      {s:28s}: {c:4d}")

    overall = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"\n  OVERALL: {overall}/{total} = {overall/total*100:.1f}%")

    # Save
    report = {
        "total": total, "correct": overall, "accuracy": round(overall/total, 4),
        "pathvqa": {"total": 100, "correct": pvqa_ok, "time_s": round(elapsed_pvqa, 1)},
        "vqarad": {"total": 100, "correct": vqarad_ok, "time_s": round(elapsed_vqarad, 1)},
        "pubmedqa": {"total": 200, "correct": pubmed_ok, "time_s": round(elapsed_pubmed, 1)},
    }
    out = ROOT / "benchmarks" / "results" / "real_image_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run())
