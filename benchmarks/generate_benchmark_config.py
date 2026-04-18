#!/usr/bin/env python3
"""Auto-generate router config from benchmark model pool.

Reads benchmark ground-truth data files to discover the model pool
and pricing, then writes a config YAML that the router can consume.
Decision exemplars (routing intelligence) are loaded from a shared
base so only the model pool changes per benchmark.

Usage:
  python generate_benchmark_config.py --benchmark routerarena \
      --data-dir ../RouterArena/cached_results/ \
      --cost-file ../RouterArena/model_cost/model_cost.json

  python generate_benchmark_config.py --benchmark vl-routerbench \
      --data-dir benchmarks/vl_routerbench/

  # Programmatic:
  from generate_benchmark_config import generate_config
  config_path = generate_config(
      benchmark="routerarena",
      data_dir="../RouterArena/cached_results/",
      cost_file="../RouterArena/model_cost/model_cost.json",
  )
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Shared decision exemplars (routing intelligence)
# ---------------------------------------------------------------------------

SHARED_DECISIONS: list[dict[str, Any]] = [
    {
        "name": "complex_reasoning",
        "description": "Complex math, proofs, logic, multi-step analysis",
        "exemplars": [
            "prove that the square root of 2 is irrational",
            "derive the quadratic formula from first principles",
            "solve this system of differential equations",
            "find the eigenvalues and eigenvectors of matrix A",
            "prove by induction that the sum of first n integers equals n(n+1)/2",
            "analyze the time complexity of this algorithm",
            "what is the limit of this sequence as n approaches infinity",
            "explain why the halting problem is undecidable",
            "prove the fundamental theorem of calculus",
            "solve this optimization problem using Lagrange multipliers",
        ],
        "require": ["reasoning"],
        "strategy": "quality_first",
        "config": {"temperature": 0, "max_tokens": 4096},
        "min_similarity": 0.45,
    },
    {
        "name": "code_generation",
        "description": "Writing, debugging, reviewing, or explaining code",
        "exemplars": [
            "write a Python function to sort a list",
            "debug this JavaScript code",
            "implement a binary search tree in Java",
            "write a SQL query to find duplicate records",
            "create a REST API endpoint in FastAPI",
            "explain what this regex does",
            "refactor this function to be more efficient",
            "write unit tests for this class",
            "implement merge sort in C++",
            "fix the bug in this Python script",
        ],
        "require": ["code"],
        "strategy": "cheapest_capable",
        "config": {"temperature": 0.2, "max_tokens": 2048},
        "min_similarity": 0.4,
    },
    {
        "name": "creative_writing",
        "description": "Stories, poems, creative content",
        "exemplars": [
            "write a short story about a time traveler",
            "compose a haiku about autumn",
            "write a compelling product description",
            "create a dialogue between two historical figures",
            "write a poem about the ocean",
            "write a persuasive essay about climate change",
        ],
        "require": ["text"],
        "strategy": "cheapest_capable",
        "config": {"temperature": 0.9, "max_tokens": 2048},
        "min_similarity": 0.35,
    },
    {
        "name": "simple_qa",
        "description": "Simple factual questions, trivia, definitions",
        "exemplars": [
            "what is the capital of France",
            "who invented the telephone",
            "how many planets are in the solar system",
            "what year did World War 2 end",
            "define photosynthesis",
            "what is the boiling point of water",
            "who wrote Romeo and Juliet",
            "what is the largest ocean on earth",
        ],
        "require": ["text"],
        "strategy": "cheapest_capable",
        "config": {"temperature": 0.1, "max_tokens": 200},
        "min_similarity": 0.35,
    },
    {
        "name": "general",
        "description": "General conversation, summarization, rewriting",
        "exemplars": [
            "tell me about yourself",
            "what can you help me with",
            "summarize this article",
            "rewrite this email to be more professional",
            "explain this concept in simple terms",
            "give me advice on how to study effectively",
            "compare these two approaches",
            "help me plan my vacation",
        ],
        "require": ["knowledge"],
        "strategy": "cheapest_capable",
        "config": {"temperature": 0.7, "max_tokens": 1024},
        "min_similarity": 0.25,
    },
]

SHARED_SAFETY_RULES: list[dict[str, Any]] = [
    {
        "name": "jailbreak_block",
        "if": "safety > 0.7",
        "action": "block",
        "reason": "Safety risk detected",
    },
]

SHARED_LEGACY_RULES: list[dict[str, Any]] = [
    {
        "default": {
            "require": ["knowledge"],
            "strategy": "cheapest_capable",
            "config": {"temperature": 0.7},
            "reason": "General query, cheapest knowledge-capable model",
        },
    },
]


# ---------------------------------------------------------------------------
# Provider / capability heuristics
# ---------------------------------------------------------------------------

def _infer_provider(model_name: str) -> str:
    """Infer API provider from model name."""
    name = model_name.lower()
    if "gpt" in name or "o1" in name or "o3" in name or "o4" in name:
        return "openai"
    if "claude" in name:
        return "anthropic"
    if "gemini" in name or "gemma" in name:
        return "google"
    if "qwen" in name:
        return "alibaba"
    if "deepseek" in name:
        return "deepseek"
    if "glm" in name:
        return "zhipu"
    return "unknown"


def _infer_capabilities(model_name: str) -> list[str]:
    """Infer model capabilities from its name."""
    name = model_name.lower()
    caps = ["text", "general"]
    if any(k in name for k in ("gpt-4o", "gemini", "claude-3")):
        caps.append("knowledge")
    if any(k in name for k in ("gpt-4o", "claude-3.5", "claude-sonnet")):
        caps.append("reasoning")
    if any(k in name for k in ("gpt-4", "claude", "deepseek", "qwen")):
        caps.append("code")
    if "flash" in name or "mini" in name or "nano" in name or "haiku" in name:
        caps.append("fast")
    return list(dict.fromkeys(caps))  # deduplicate, preserve order


def _infer_quality(model_name: str) -> float:
    """Assign a default quality score based on model tier heuristics."""
    name = model_name.lower()
    if any(k in name for k in ("gpt-4o-mini", "flash", "haiku", "nano")):
        return 0.70
    if any(k in name for k in ("gpt-4o", "claude-3-opus", "sonnet")):
        return 0.90
    return 0.75


def _infer_latency(model_name: str) -> float:
    """Estimate average latency in ms."""
    name = model_name.lower()
    if any(k in name for k in ("flash", "mini", "nano", "haiku")):
        return 250
    if any(k in name for k in ("opus", "pro")):
        return 800
    return 400


# ---------------------------------------------------------------------------
# RouterArena config generator
# ---------------------------------------------------------------------------

def generate_from_routerarena(
    data_dir: str,
    cost_file: Optional[str] = None,
) -> dict[str, Any]:
    """Read RouterArena cached_results/ to discover models and build config.

    Args:
        data_dir: Path to cached_results/ containing per-model .jsonl files.
        cost_file: Optional path to model_cost.json for official pricing.

    Returns:
        Config dict ready to be dumped as YAML.
    """
    data_path = Path(data_dir)
    jsonl_files = sorted(glob(str(data_path / "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    # ----- Read pricing data -----
    pricing: dict[str, dict] = {}
    if cost_file and Path(cost_file).exists():
        with open(cost_file) as f:
            pricing = json.load(f)
    else:
        # Try conventional location
        default_cost = data_path.parent / "model_cost" / "model_cost.json"
        if default_cost.exists():
            with open(default_cost) as f:
                pricing = json.load(f)

    # ----- Discover models from data files -----
    models_info: dict[str, dict[str, Any]] = {}
    for fpath in jsonl_files:
        model_name = Path(fpath).stem
        with open(fpath) as fh:
            lines = fh.readlines()

        query_count = len(lines)

        # Sample first entry for token usage and cost
        first = json.loads(lines[0])
        sample_cost = first.get("evaluation_result", {}).get("inference_cost", 0)
        token_usage = first.get("token_usage", {})
        avg_input = token_usage.get("input_tokens", 500)
        avg_output = token_usage.get("output_tokens", 300)

        # Get pricing -- prefer model_cost.json, fall back to computed
        model_pricing = pricing.get(model_name, {})
        cost_per_1k_input = model_pricing.get(
            "input_token_price_per_million", None
        )
        cost_per_1k_output = model_pricing.get(
            "output_token_price_per_million", None
        )

        # Convert $/M tokens -> $/1K tokens
        if cost_per_1k_input is not None:
            cost_per_1k_input = cost_per_1k_input / 1000.0
        else:
            # Estimate from sample cost and token counts
            total_tokens = avg_input + avg_output
            if total_tokens > 0 and sample_cost > 0:
                cost_per_token = sample_cost / total_tokens
                cost_per_1k_input = cost_per_token * 1000
                cost_per_1k_output = cost_per_token * 1000
            else:
                cost_per_1k_input = 0.15
                cost_per_1k_output = 0.60

        if cost_per_1k_output is not None and isinstance(
            model_pricing.get("output_token_price_per_million"), (int, float)
        ):
            cost_per_1k_output = model_pricing["output_token_price_per_million"] / 1000.0

        models_info[model_name] = {
            "name": model_name,
            "query_count": query_count,
            "sample_cost": sample_cost,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "cost_per_1k_input": round(cost_per_1k_input, 6),
            "cost_per_1k_output": round(cost_per_1k_output, 6),
        }

    # ----- Build config -----
    model_configs = []
    for name, info in sorted(models_info.items(), key=lambda x: x[1]["cost_per_1k_input"]):
        model_configs.append({
            "name": name,
            "provider": _infer_provider(name),
            "cost_per_1k_input": info["cost_per_1k_input"],
            "cost_per_1k_output": info["cost_per_1k_output"],
            "avg_latency_ms": _infer_latency(name),
            "capabilities": _infer_capabilities(name),
            "quality_score": _infer_quality(name),
        })

    config: dict[str, Any] = {
        "models": model_configs,
        "routing": {
            "budget": {
                "max_cost_per_query": 0.01,
                "strategy": "cheapest_capable",
                "quality_threshold": 0.7,
            },
            "decisions": SHARED_DECISIONS,
            "safety_rules": SHARED_SAFETY_RULES,
            "rules": SHARED_LEGACY_RULES,
        },
    }

    # Add metadata comment via a top-level key
    config["_auto_generated"] = {
        "benchmark": "routerarena",
        "data_dir": str(data_path.resolve()),
        "model_count": len(model_configs),
        "total_queries": sum(m["query_count"] for m in models_info.values()),
        "models_discovered": list(models_info.keys()),
    }

    return config


# ---------------------------------------------------------------------------
# VL-RouterBench config generator
# ---------------------------------------------------------------------------

def generate_from_vl_routerbench(data_dir: str) -> dict[str, Any]:
    """Read VL-RouterBench data to discover VLM pool and build config.

    Args:
        data_dir: Path to benchmarks/vl_routerbench/ directory.

    Returns:
        Config dict ready to be dumped as YAML.
    """
    data_path = Path(data_dir)

    # Read models list
    models_file = data_path / "data" / "registry" / "models.txt"
    if not models_file.exists():
        raise FileNotFoundError(f"models.txt not found at {models_file}")

    with open(models_file) as f:
        model_names = [line.strip() for line in f if line.strip()]

    # Read models.json for cost info
    models_json_file = data_path / "data" / "registry" / "models.json"
    models_json_list: list[dict] = []
    if models_json_file.exists():
        with open(models_json_file) as f:
            raw = json.load(f)
            # Handle both {"models": [...]} and [...] formats
            if isinstance(raw, dict) and "models" in raw:
                models_json_list = raw["models"]
            elif isinstance(raw, list):
                models_json_list = raw
            else:
                models_json_list = []

    # Index models.json by name for lookup
    models_json_by_name: dict[str, dict] = {
        m["name"]: m for m in models_json_list if "name" in m
    }

    # Build model configs from discovered VLMs
    model_configs = []
    for i, name in enumerate(model_names):
        # Get cost info from models.json if available
        cost_info = models_json_by_name.get(name, {})

        input_cost = cost_info.get("cost_per_1k_input", 0.00015)
        output_cost = cost_info.get("cost_per_1k_output", 0.0006)

        # VL models.json already has cost_per_1k (not per_million)
        model_configs.append({
            "name": name,
            "provider": _infer_provider(name),
            "cost_per_1k_input": round(input_cost, 6),
            "cost_per_1k_output": round(output_cost, 6),
            "avg_latency_ms": _infer_latency(name),
            "capabilities": ["vision", "text", "general"] + cost_info.get("strengths", []),
            "quality_score": _infer_quality(name),
        })

    # VL-specific decisions
    vl_decisions = list(SHARED_DECISIONS) + [
        {
            "name": "vision_task",
            "description": "Image understanding, chart reading, visual QA",
            "exemplars": [
                "what is shown in this image",
                "describe the chart and extract the data",
                "read the text in this screenshot",
                "analyze this diagram",
                "what breed is this dog in the photo",
                "compare these two images",
            ],
            "require": ["vision"],
            "strategy": "cheapest_capable",
            "config": {"temperature": 0.3, "max_tokens": 1024},
            "min_similarity": 0.35,
        },
    ]

    config: dict[str, Any] = {
        "models": model_configs,
        "routing": {
            "budget": {
                "max_cost_per_query": 0.01,
                "strategy": "cheapest_capable",
                "quality_threshold": 0.7,
            },
            "decisions": vl_decisions,
            "safety_rules": SHARED_SAFETY_RULES,
            "rules": SHARED_LEGACY_RULES,
        },
        "_auto_generated": {
            "benchmark": "vl-routerbench",
            "data_dir": str(data_path.resolve()),
            "model_count": len(model_configs),
            "models_discovered": model_names,
        },
    }

    return config


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

GENERATORS = {
    "routerarena": generate_from_routerarena,
    "vl-routerbench": generate_from_vl_routerbench,
    "vl_routerbench": generate_from_vl_routerbench,
}


def generate_config(
    benchmark: str,
    data_dir: str,
    cost_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Generate a benchmark config and write it to configs/.

    Args:
        benchmark: Benchmark name (routerarena, vl-routerbench).
        data_dir: Path to the benchmark data directory.
        cost_file: Optional path to model_cost.json (RouterArena only).
        output_dir: Where to write the config. Defaults to configs/ in this dir.

    Returns:
        Path to the generated config YAML file.
    """
    benchmark_key = benchmark.lower().replace("-", "_")
    generator = GENERATORS.get(benchmark_key) or GENERATORS.get(benchmark.lower())
    if generator is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark!r}. "
            f"Supported: {list(GENERATORS.keys())}"
        )

    # Generate config dict
    if benchmark_key == "routerarena":
        config = generator(data_dir, cost_file)
    else:
        config = generator(data_dir)

    # Write to file
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "config" / "generated")
    os.makedirs(output_dir, exist_ok=True)

    # Normalize filename
    config_name = benchmark.lower().replace("-", "_")
    output_path = os.path.join(output_dir, f"{config_name}.yaml")

    with open(output_path, "w") as f:
        # Write header comment
        meta = config.pop("_auto_generated", {})
        f.write(f"# Auto-generated config for {benchmark}\n")
        f.write(f"# Models: {meta.get('model_count', '?')}\n")
        f.write(f"# Source: {meta.get('data_dir', data_dir)}\n")
        if "total_queries" in meta:
            f.write(f"# Total queries in benchmark: {meta['total_queries']}\n")
        f.write(f"# Models discovered: {meta.get('models_discovered', [])}\n")
        f.write("#\n# Do not edit manually -- re-run generate_benchmark_config.py instead.\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[generate_benchmark_config] Wrote {output_path}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Models: {meta.get('model_count', '?')}")
    print(f"  Discovered: {meta.get('models_discovered', [])}")

    # Validate: print model pool summary
    for m in config.get("models", []):
        print(f"    {m['name']:<35} in=${m['cost_per_1k_input']:.4f}/1K  "
              f"out=${m['cost_per_1k_output']:.4f}/1K  "
              f"quality={m['quality_score']}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate router config from benchmark data",
    )
    parser.add_argument(
        "--benchmark", "-b",
        required=True,
        choices=["routerarena", "vl-routerbench", "vl_routerbench"],
        help="Which benchmark to generate config for",
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to benchmark data directory",
    )
    parser.add_argument(
        "--cost-file", "-c",
        default=None,
        help="Path to model_cost.json (RouterArena only)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Where to write configs (default: configs/)",
    )
    args = parser.parse_args()

    config_path = generate_config(
        benchmark=args.benchmark,
        data_dir=args.data_dir,
        cost_file=args.cost_file,
        output_dir=args.output_dir,
    )
    print(f"\nConfig ready: {config_path}")


if __name__ == "__main__":
    main()
