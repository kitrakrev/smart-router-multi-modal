"""Benchmark the LLM Router against RouterArena ground truth data.

Auto-discovers models from cached_results/*.jsonl files -- ZERO hardcoded
model names. Config is auto-generated from the benchmark data via
generate_benchmark_config.py.

For overlap queries (all models have scores):
  -> Router picks a model; correct if picked model's score >= best score.

For primary-only queries (only one model has ground truth):
  -> Router must pick that model to be counted as correct.

Measures:
  - Routing accuracy
  - Cost per 1K queries
  - Model distribution
  - Average routing latency
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import run_all_signals
from src.router import Router
from benchmarks.generate_benchmark_config import generate_config


# ---------------------------------------------------------------------------
# Config -- auto-generated from benchmark data, ZERO hardcoded model names
# ---------------------------------------------------------------------------

CACHED_RESULTS_DIR = os.environ.get(
    "ROUTERARENA_CACHED",
    str(Path(__file__).parent.parent / "RouterArena" / "cached_results"),
)

COST_FILE = os.environ.get(
    "ROUTERARENA_COST_FILE",
    str(Path(__file__).parent.parent / "RouterArena" / "model_cost" / "model_cost.json"),
)


def _build_cost_table(config_path: str) -> dict[str, float]:
    """Read the auto-generated config to build a cost-per-query table.

    Uses avg 500 input + 300 output tokens per query (same assumption
    as the original hand-written config).
    """
    import yaml as _yaml
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    costs = {}
    for m in cfg.get("models", []):
        name = m["name"]
        cost = (500 / 1000) * m["cost_per_1k_input"] + (300 / 1000) * m["cost_per_1k_output"]
        costs[name] = cost
    costs["BLOCKED"] = 0.0
    return costs


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> dict[str, dict]:
    """Load JSONL file into dict keyed by global_index."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                data[entry["global_index"]] = entry
    return data


def load_all_ground_truth(cached_dir: str):
    """Load all cached result files and merge into a unified structure.

    Discovers models from .jsonl filenames -- ZERO hardcoded model names.

    Returns:
        all_queries: list of dicts with keys:
            global_index, question, metric, scores: {model: score}, has_all_models: bool
        overlap_count: number of queries where all models have scores
        model_names: list of discovered model names
    """
    from glob import glob as _glob

    jsonl_files = sorted(_glob(os.path.join(cached_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {cached_dir}")

    # Load all model data
    model_data: dict[str, dict] = {}
    for fpath in jsonl_files:
        model_name = Path(fpath).stem
        model_data[model_name] = load_jsonl(fpath)

    model_names = list(model_data.keys())

    # Find the model with the most queries (primary)
    primary_model = max(model_names, key=lambda m: len(model_data[m]))
    other_models = [m for m in model_names if m != primary_model]

    # Overlap = indices present in ALL models
    if other_models:
        overlap_indices = set.intersection(*(set(model_data[m].keys()) for m in other_models))
    else:
        overlap_indices = set(model_data[primary_model].keys())

    all_queries = []
    for idx, entry in model_data[primary_model].items():
        question = entry.get("question", "")
        metric = entry.get("evaluation_result", {}).get("metric", "unknown")
        primary_score = entry.get("evaluation_result", {}).get("score", 0.0)

        scores = {primary_model: primary_score}
        has_all = idx in overlap_indices

        if has_all:
            for other in other_models:
                other_entry = model_data[other].get(idx, {})
                scores[other] = other_entry.get("evaluation_result", {}).get("score", 0.0)

        all_queries.append({
            "global_index": idx,
            "question": question,
            "metric": metric,
            "scores": scores,
            "has_all_models": has_all,
        })

    return all_queries, len(overlap_indices), model_names


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

async def benchmark():
    print("=" * 70)
    print("  LLM Router Benchmark vs RouterArena Ground Truth")
    print("  (auto-generated config from benchmark data)")
    print("=" * 70)

    # Auto-generate config from benchmark data
    print(f"\nAuto-generating config from benchmark data...")
    config_path = generate_config(
        benchmark="routerarena",
        data_dir=CACHED_RESULTS_DIR,
        cost_file=COST_FILE,
    )

    # Build cost table from auto-generated config (no hardcoded costs)
    MODEL_COST_PER_QUERY = _build_cost_table(config_path)

    # Load data
    print(f"\nLoading ground truth from: {CACHED_RESULTS_DIR}")
    all_queries, overlap_count, discovered_models = load_all_ground_truth(CACHED_RESULTS_DIR)
    primary_only_count = len(all_queries) - overlap_count
    print(f"Total queries: {len(all_queries)}")
    print(f"  Overlap ({len(discovered_models)}-model): {overlap_count}")
    print(f"  Primary-only:      {primary_only_count}")
    print(f"  Models discovered: {discovered_models}")

    # Validate: all models in config have ground truth
    router = Router(config_path=config_path)
    config_models = set(router.models.keys())
    gt_models = set(discovered_models)
    if config_models != gt_models:
        missing_gt = config_models - gt_models
        extra_gt = gt_models - config_models
        if missing_gt:
            print(f"  WARNING: models in config but not in ground truth: {missing_gt}")
        if extra_gt:
            print(f"  WARNING: models in ground truth but not in config: {extra_gt}")
    else:
        print(f"  VALIDATED: all {len(config_models)} config models have ground truth")

    print(f"\nRouter loaded with models: {list(router.models.keys())}")
    print(f"Routing rules: {len(router.rules)}")

    # Stats
    total = len(all_queries)
    latencies: list[float] = []
    model_picks: Counter = Counter()
    decision_picks: Counter = Counter()
    similarity_scores: list[float] = []
    correct_picks = 0
    optimal_picks = 0  # picked the actual best model
    total_cost = 0.0
    blocked_count = 0
    score_by_model: defaultdict[str, list[float]] = defaultdict(list)
    metric_counts: Counter = Counter()

    # Determine the primary model (one with most queries)
    primary_model = max(discovered_models, key=lambda m: sum(
        1 for q in all_queries if m in q["scores"]
    ))

    # Breakdown stats
    overlap_correct = 0
    overlap_total = 0
    primary_only_correct = 0
    primary_only_total = 0

    # Progress
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(all_queries), total=total, desc="Benchmarking", unit="query")
    except ImportError:
        print("\n[tqdm not installed, showing progress every 500 queries]")
        iterator = enumerate(all_queries)

    for i, entry in iterator:
        question = entry["question"]
        scores = entry["scores"]
        metric = entry["metric"]
        has_all = entry["has_all_models"]

        metric_counts[metric] += 1

        messages = [{"role": "user", "content": question}]

        # Run signals + routing
        t0 = time.perf_counter()
        signals = await run_all_signals(messages)
        decision = router.decide(signals, query_text=question)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        selected = decision.selected_model
        model_picks[selected] += 1
        decision_picks[decision.decision_name or "unknown"] += 1
        if decision.similarity > 0:
            similarity_scores.append(decision.similarity)

        if decision.blocked:
            blocked_count += 1
            continue

        # Cost accounting
        query_cost = MODEL_COST_PER_QUERY.get(selected, 0.0)
        total_cost += query_cost

        # --- Accuracy evaluation ---
        if has_all:
            overlap_total += 1
            # All 3 models have scores. Router is correct if the selected
            # model's score equals the best available score.
            best_score = max(scores.values())
            selected_score = scores.get(selected, -1.0)

            if selected_score >= best_score:
                correct_picks += 1
                overlap_correct += 1
                optimal_picks += 1
            elif selected in scores and selected_score >= 0:
                # Model exists in ground truth but didn't get best score.
                # Still "partially" ok but not counted as correct.
                pass
            # else: selected model not in ground truth at all -> wrong

            if selected in scores:
                score_by_model[selected].append(selected_score)
            else:
                score_by_model[selected].append(0.0)
        else:
            primary_only_total += 1
            # Only the primary model has ground truth.
            if selected == primary_model:
                correct_picks += 1
                primary_only_correct += 1
                optimal_picks += 1
                score_by_model[selected].append(scores[primary_model])
            else:
                # We don't know this model's score -> can't verify -> wrong
                score_by_model[selected].append(0.0)

        # Progress for non-tqdm
        if not hasattr(iterator, 'set_postfix') and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total} queries "
                  f"(avg latency: {sum(latencies) / len(latencies):.2f}ms)")

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------

    evaluated = total - blocked_count
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    cost_per_1k = (total_cost / max(evaluated, 1)) * 1000

    gt_cost_per_1k = MODEL_COST_PER_QUERY[primary_model] * 1000

    accuracy = correct_picks / max(evaluated, 1)
    optimal_rate = optimal_picks / max(evaluated, 1)

    overlap_acc = overlap_correct / max(overlap_total, 1)
    primary_only_acc = primary_only_correct / max(primary_only_total, 1)

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'Value':>15}")
    print("-" * 57)
    print(f"{'Total queries':<40} {total:>15,}")
    print(f"{'Blocked (safety)':<40} {blocked_count:>15,}")
    print(f"{'Evaluated':<40} {evaluated:>15,}")
    print(f"{'Overall routing accuracy':<40} {accuracy:>14.1%}")
    print(f"{'  Overlap (3-model) accuracy':<40} {overlap_acc:>14.1%}")
    print(f"{'  Primary-only accuracy':<40} {primary_only_acc:>14.1%}")
    print(f"{'Optimal model selection':<40} {optimal_rate:>14.1%}")
    print(f"{'Cost per 1K queries (router)':<40} {'$' + f'{cost_per_1k:.4f}':>15}")
    print(f"{'Cost per 1K queries (all ' + primary_model + ')':<40} {'$' + f'{gt_cost_per_1k:.4f}':>15}")
    print(f"{'Cost ratio vs all-mini':<40} {cost_per_1k / gt_cost_per_1k:>14.2f}x")
    print(f"{'Avg routing latency':<40} {avg_latency:>12.2f} ms")
    print(f"{'P50 routing latency':<40} {p50_latency:>12.2f} ms")
    print(f"{'P99 routing latency':<40} {p99_latency:>12.2f} ms")

    print(f"\n{'Model Distribution':}")
    print("-" * 57)
    for model, count in model_picks.most_common():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<30} {count:>6} ({pct:5.1f}%) {bar}")

    print(f"\n{'Decision Distribution (embedding-matched)':}")
    print("-" * 57)
    for dec, count in decision_picks.most_common():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {dec:<30} {count:>6} ({pct:5.1f}%) {bar}")

    if similarity_scores:
        avg_sim = sum(similarity_scores) / len(similarity_scores)
        min_sim = min(similarity_scores)
        max_sim = max(similarity_scores)
        print(f"\n{'Embedding Similarity Stats':}")
        print("-" * 57)
        print(f"  {'Queries with similarity > 0':<30} {len(similarity_scores):>6}")
        print(f"  {'Average similarity':<30} {avg_sim:>9.3f}")
        print(f"  {'Min similarity':<30} {min_sim:>9.3f}")
        print(f"  {'Max similarity':<30} {max_sim:>9.3f}")

    print(f"\n{'Metric Distribution (ground truth)':}")
    print("-" * 57)
    for metric, count in metric_counts.most_common():
        print(f"  {metric:<30} {count:>6}")

    print(f"\n{'Ground Truth Score by Router Pick':}")
    print("-" * 57)
    for model in sorted(score_by_model.keys()):
        scores_list = score_by_model[model]
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
        print(f"  {model:<30} avg_gt_score={avg_score:.3f}  n={len(scores_list)}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    results = {
        "benchmark": "RouterArena",
        "config": config_path,
        "total_queries": total,
        "blocked": blocked_count,
        "evaluated": evaluated,
        "routing_accuracy": round(accuracy, 4),
        "overlap_accuracy": round(overlap_acc, 4),
        "primary_only_accuracy": round(primary_only_acc, 4),
        "optimal_selection_rate": round(optimal_rate, 4),
        "cost_per_1k_router": round(cost_per_1k, 4),
        "cost_per_1k_baseline": round(gt_cost_per_1k, 4),
        "cost_ratio": round(cost_per_1k / gt_cost_per_1k, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "model_distribution": dict(model_picks),
        "decision_distribution": dict(decision_picks),
        "avg_similarity": round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0.0,
        "metric_distribution": dict(metric_counts),
        "gt_score_by_model": {
            m: {"avg_score": round(sum(s) / len(s), 4) if s else 0, "count": len(s)}
            for m, s in score_by_model.items()
        },
    }

    output_path = Path(__file__).parent / "results" / "routerarena.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("  RouterArena Leaderboard Format")
    print("=" * 70)
    print(f"  {'Router':<30} {'Accuracy':>10} {'Cost/1K':>12} {'Latency':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10}")
    print(f"  {'LLM Router (fixed)':<30} {accuracy:>9.1%} {'$'+f'{cost_per_1k:.4f}':>12} {avg_latency:>8.2f}ms")
    print(f"  {'All ' + primary_model:<30} {'—':>10} {'$'+f'{gt_cost_per_1k:.4f}':>12} {'—':>10}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Before vs After comparison
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER FIX")
    print("=" * 70)
    before = {
        "routing_accuracy": 0.2609,
        "cost_per_1k_router": 857.6876,
        "cost_ratio": 3.3635,
        "model_distribution": {
            "claude-sonnet": 1109, "qwen-3b-local": 5222,
            "gpt-4o-mini": 2043, "BLOCKED": 26,
        },
    }
    print(f"\n  {'Metric':<35} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    before_acc = before['routing_accuracy']
    before_cost = before['cost_per_1k_router']
    before_ratio = before['cost_ratio']
    print(f"  {'Routing accuracy':<35} {before_acc:>11.1%} {accuracy:>11.1%} {accuracy - before_acc:>+11.1%}")
    cost_before_str = f"${before_cost:.2f}"
    cost_after_str = f"${cost_per_1k:.2f}"
    cost_delta_str = f"${cost_per_1k - before_cost:.2f}"
    print(f"  {'Cost per 1K queries':<35} {cost_before_str:>12} {cost_after_str:>12} {cost_delta_str:>12}")
    after_ratio = cost_per_1k / gt_cost_per_1k
    print(f"  {'Cost ratio vs all-mini':<35} {before_ratio:>11.2f}x {after_ratio:>11.2f}x {'':>12}")
    print(f"\n  Model distribution change:")
    print(f"  {'Model':<30} {'Before':>8} {'After':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    all_models_set = set(before['model_distribution'].keys()) | set(model_picks.keys())
    for m in sorted(all_models_set):
        b = before['model_distribution'].get(m, 0)
        a = model_picks.get(m, 0)
        print(f"  {m:<30} {b:>8} {a:>8}")

    print(f"\n  Root cause: old config routed to claude-sonnet and qwen-3b-local")
    print(f"  which don't exist in RouterArena's 3-model pool.")
    print(f"  Fix: config_routerarena.yaml uses only the 3 RouterArena models.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(benchmark())
