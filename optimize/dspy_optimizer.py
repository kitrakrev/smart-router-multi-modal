#!/usr/bin/env python3
"""DSPy prompt optimization per model x specialty pair.

For each (model, specialty):
  1. Load dataset (PathVQA for pathology, PubMedQA for clinical, etc.)
  2. Define DSPy Signature (query + image_desc -> answer)
  3. Run MIPROv2 optimizer with accuracy_metric
  4. Save optimized prompt to config/prompt_templates.yaml overrides section

CLI:
  python -m optimize.dspy_optimizer --model medgemma-4b --specialty pathology --dataset pathvqa
  python -m optimize.dspy_optimizer --all  # optimize all pairs

Uses small subsets: 100 train, 50 eval (DSPy doesn't need much).
Saves results as YAML with metadata: accuracy, dataset, timestamp.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import dspy
    from dspy import Signature, InputField, OutputField, ChainOfThought
    from dspy.teleprompt import MIPROv2
except ImportError:
    dspy = None

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROMPT_TEMPLATES_PATH = CONFIG_DIR / "prompt_templates.yaml"
RESULTS_DIR = ROOT / "optimize" / "results"

# ---------------------------------------------------------------------------
# Dataset loaders (partial — small subsets for DSPy)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "pathvqa": {
        "hf_name": "flaviagiammarino/path-vqa",
        "hf_config": None,
        "split": "train",
        "question_key": "question",
        "answer_key": "answer",
        "context_key": None,
        "specialty": "pathology",
    },
    "pubmedqa": {
        "hf_name": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "split": "train",
        "question_key": "question",
        "answer_key": "final_decision",
        "context_key": "context",
        "specialty": "general_medicine",
    },
    "medqa": {
        "hf_name": "bigbio/med_qa",
        "hf_config": "med_qa_en_4options_bigbio_qa",
        "split": "train",
        "question_key": "question",
        "answer_key": "answer",
        "context_key": None,
        "specialty": "general_medicine",
    },
}

# Which datasets to use for each specialty
SPECIALTY_DATASETS: dict[str, list[str]] = {
    "pathology": ["pathvqa"],
    "general_medicine": ["pubmedqa", "medqa"],
    "radiology": ["pathvqa"],  # subset with radiology-like questions
    "cardiology": ["pubmedqa"],
    "dermatology": ["pathvqa"],
    "ophthalmology": ["pathvqa"],
    "emergency": ["pubmedqa"],
    "pharmacology": ["pubmedqa"],
}

# Model registry: model_id -> default API base
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "medgemma-4b": {
        "model_id": "google/medgemma-1.5-4b-it",
        "api_base": "http://localhost:8000/v1",
        "provider": "local",
    },
    "llava-med-7b": {
        "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        "api_base": "http://localhost:8000/v1",
        "provider": "local",
    },
    "medalpaca-7b": {
        "model_id": "medalpaca/medalpaca-7b",
        "api_base": "http://localhost:8000/v1",
        "provider": "local",
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "provider": "local",
    },
    "gpt-4o": {
        "model_id": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "provider": "openai",
    },
}


def load_dataset_for_dspy(
    dataset_name: str,
    max_train: int = 100,
    max_eval: int = 50,
) -> tuple[list, list]:
    """Load a partial dataset and convert to DSPy Examples.

    Returns (train_examples, eval_examples).
    """
    from datasets import load_dataset

    ds_info = DATASET_REGISTRY[dataset_name]

    print(f"  Loading {dataset_name} from HuggingFace ({ds_info['hf_name']})...")
    kwargs = {"split": ds_info["split"]}
    if ds_info["hf_config"]:
        ds = load_dataset(ds_info["hf_name"], ds_info["hf_config"], **kwargs)
    else:
        ds = load_dataset(ds_info["hf_name"], **kwargs)

    total = len(ds)
    needed = min(total, max_train + max_eval)
    ds = ds.select(range(needed))
    print(f"  Selected {needed}/{total} samples")

    examples = []
    q_key = ds_info["question_key"]
    a_key = ds_info["answer_key"]
    ctx_key = ds_info.get("context_key")

    for row in ds:
        question = str(row.get(q_key, ""))
        answer = str(row.get(a_key, ""))
        context = ""
        if ctx_key and row.get(ctx_key):
            ctx_val = row[ctx_key]
            if isinstance(ctx_val, dict):
                # PubMedQA: context is {"contexts": [...], "labels": [...]}
                contexts = ctx_val.get("contexts", [])
                context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
            elif isinstance(ctx_val, list):
                context = " ".join(str(c) for c in ctx_val)
            else:
                context = str(ctx_val)

        if not question.strip() or not answer.strip():
            continue

        if dspy is not None:
            ex = dspy.Example(
                question=question,
                context=context[:500],  # truncate long contexts
                answer=answer,
            ).with_inputs("question", "context")
        else:
            ex = {"question": question, "context": context[:500], "answer": answer}
        examples.append(ex)

    train = examples[:max_train]
    eval_set = examples[max_train : max_train + max_eval]
    print(f"  Train: {len(train)}, Eval: {len(eval_set)}")
    return train, eval_set


# ---------------------------------------------------------------------------
# DSPy Signature and scoring
# ---------------------------------------------------------------------------

if dspy is not None:

    class MedicalQA(Signature):
        """Answer a medical question given context."""

        question = InputField(desc="clinical question or VQA question")
        context = InputField(
            desc="additional context: image description, patient history", default=""
        )
        answer = OutputField(desc="medical answer")


def score_medical(prediction: str, gold: str) -> float:
    """Score a medical answer against ground truth.

    Uses a combination of:
      - Exact match (normalized)
      - Token-level F1
      - Yes/No match for PubMedQA-style answers
    """
    pred_norm = prediction.strip().lower()
    gold_norm = gold.strip().lower()

    # Exact match
    if pred_norm == gold_norm:
        return 1.0

    # Yes/No/Maybe match (PubMedQA)
    yn_map = {"yes": "yes", "no": "no", "maybe": "maybe"}
    for key in yn_map:
        if key in pred_norm[:20] and gold_norm == key:
            return 1.0

    # Token-level F1
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if not gold_tokens:
        return 0.0
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return round(f1, 4)


def evaluate(program, eval_examples: list, verbose: bool = False) -> float:
    """Run evaluation and return average accuracy."""
    if dspy is None:
        print("  [WARN] dspy not installed, returning mock accuracy")
        return 0.0

    scores = []
    for ex in eval_examples:
        try:
            pred = program(question=ex.question, context=ex.context)
            score = score_medical(pred.answer, ex.answer)
            scores.append(score)
            if verbose:
                print(f"    Q: {ex.question[:60]}...")
                print(f"    Gold: {ex.answer[:60]}  Pred: {pred.answer[:60]}  Score: {score:.2f}")
        except Exception as e:
            scores.append(0.0)
            if verbose:
                print(f"    ERROR: {e}")

    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4)


# ---------------------------------------------------------------------------
# Optimizer core
# ---------------------------------------------------------------------------


def optimize_pair(
    model_id: str,
    specialty: str,
    dataset_name: str,
    max_train: int = 100,
    max_eval: int = 50,
    num_candidates: int = 10,
    max_bootstrapped_demos: int = 3,
    api_base: str = "http://localhost:8000/v1",
) -> dict[str, Any]:
    """Optimize a single (model, specialty) pair using DSPy MIPROv2.

    Returns a dict with optimized prompt info, accuracy, and metadata.
    """
    if dspy is None:
        raise ImportError(
            "dspy is required for optimization. Install: pip install dspy-ai"
        )

    print(f"\n{'='*60}")
    print(f"  Optimizing: model={model_id} specialty={specialty} dataset={dataset_name}")
    print(f"{'='*60}")

    # 1. Load dataset
    train_examples, eval_examples = load_dataset_for_dspy(
        dataset_name, max_train=max_train, max_eval=max_eval
    )

    if not train_examples:
        return {"error": f"No training examples loaded from {dataset_name}"}

    # 2. Configure DSPy with target model
    lm = dspy.LM(model_id, api_base=api_base)
    dspy.configure(lm=lm)

    # 3. Create program
    program = ChainOfThought(MedicalQA)

    # 4. Pre-optimization baseline
    print("\n  Running baseline evaluation...")
    baseline_acc = evaluate(program, eval_examples[:20])
    print(f"  Baseline accuracy (20 samples): {baseline_acc:.4f}")

    # 5. Run MIPROv2 optimizer
    print("\n  Running MIPROv2 optimization...")

    def metric_fn(example, pred, trace=None):
        return score_medical(pred.answer, example.answer)

    optimizer = MIPROv2(
        metric=metric_fn,
        num_candidates=num_candidates,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )

    optimized = optimizer.compile(program, trainset=train_examples)

    # 6. Post-optimization evaluation
    print("\n  Running optimized evaluation...")
    optimized_acc = evaluate(optimized, eval_examples)
    print(f"  Optimized accuracy: {optimized_acc:.4f}")

    # 7. Extract optimized prompt and demos
    instructions = ""
    demos = []
    try:
        if hasattr(optimized, "predict") and hasattr(optimized.predict, "signature"):
            instructions = str(optimized.predict.signature.instructions)
        elif hasattr(optimized, "signature"):
            instructions = str(optimized.signature.instructions)

        if hasattr(optimized, "demos"):
            demos = [
                {"question": d.question, "answer": d.answer}
                for d in optimized.demos
                if hasattr(d, "question") and hasattr(d, "answer")
            ]
    except Exception as e:
        print(f"  [WARN] Could not extract prompt details: {e}")

    result = {
        "model": model_id,
        "specialty": specialty,
        "dataset": dataset_name,
        "system_prompt": instructions,
        "few_shot_demos": demos,
        "baseline_accuracy": baseline_acc,
        "optimized_accuracy": optimized_acc,
        "improvement": round(optimized_acc - baseline_acc, 4),
        "num_candidates": num_candidates,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_result(result: dict, output_dir: Path | None = None):
    """Save optimization result to YAML and update prompt_templates overrides."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual result
    model_safe = result.get("model", "unknown").replace("/", "_")
    specialty = result.get("specialty", "unknown")
    filename = f"{model_safe}_{specialty}_{result.get('dataset', 'unknown')}.yaml"
    result_path = output_dir / filename

    with open(result_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    print(f"  Result saved to: {result_path}")

    # Update prompt_templates.yaml overrides
    if result.get("system_prompt") or result.get("few_shot_demos"):
        _update_prompt_overrides(result)


def _update_prompt_overrides(result: dict):
    """Merge optimization result into config/prompt_templates.yaml overrides."""
    if not PROMPT_TEMPLATES_PATH.exists():
        print(f"  [WARN] {PROMPT_TEMPLATES_PATH} not found, skipping override update")
        return

    with open(PROMPT_TEMPLATES_PATH, "r") as f:
        templates = yaml.safe_load(f) or {}

    overrides = templates.setdefault("overrides", {})
    model_key = result.get("model", "unknown")
    specialty_key = result.get("specialty", "unknown")

    override_key = f"{model_key}_{specialty_key}"
    overrides[override_key] = {
        "system_prompt": result.get("system_prompt", ""),
        "few_shot_demos": result.get("few_shot_demos", []),
        "accuracy": result.get("optimized_accuracy", 0.0),
        "dataset": result.get("dataset", ""),
        "timestamp": result.get("timestamp", ""),
    }

    with open(PROMPT_TEMPLATES_PATH, "w") as f:
        yaml.dump(templates, f, default_flow_style=False, sort_keys=False)
    print(f"  Updated overrides in {PROMPT_TEMPLATES_PATH}: {override_key}")


# ---------------------------------------------------------------------------
# All-pairs optimization
# ---------------------------------------------------------------------------

ALL_PAIRS: list[tuple[str, str, str]] = [
    # (model_id, specialty, dataset)
    ("medgemma-4b", "pathology", "pathvqa"),
    ("medgemma-4b", "general_medicine", "pubmedqa"),
    ("llava-med-7b", "pathology", "pathvqa"),
    ("medalpaca-7b", "general_medicine", "pubmedqa"),
    ("llama-3.1-8b", "general_medicine", "pubmedqa"),
]


def optimize_all(api_base: str = "http://localhost:8000/v1") -> list[dict]:
    """Optimize all configured (model, specialty, dataset) triples."""
    results = []
    for model_id, specialty, dataset_name in ALL_PAIRS:
        model_info = MODEL_REGISTRY.get(model_id, {})
        base = model_info.get("api_base", api_base)
        actual_model = model_info.get("model_id", model_id)

        try:
            result = optimize_pair(
                model_id=actual_model,
                specialty=specialty,
                dataset_name=dataset_name,
                api_base=base,
            )
            save_result(result)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed {model_id}/{specialty}/{dataset_name}: {e}")
            results.append({
                "model": model_id,
                "specialty": specialty,
                "dataset": dataset_name,
                "error": str(e),
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  FAIL  {r['model']}/{r['specialty']}: {r['error'][:60]}")
        else:
            print(
                f"  OK    {r['model']}/{r['specialty']}: "
                f"baseline={r['baseline_accuracy']:.3f} -> "
                f"optimized={r['optimized_accuracy']:.3f} "
                f"(+{r['improvement']:.3f})"
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DSPy prompt optimization for MedVisionRouter"
    )
    parser.add_argument(
        "--model", type=str, help="Model ID (e.g., medgemma-4b)"
    )
    parser.add_argument(
        "--specialty", type=str, help="Medical specialty (e.g., pathology)"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name (e.g., pathvqa, pubmedqa)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Optimize all configured pairs"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for the model server",
    )
    parser.add_argument(
        "--max-train", type=int, default=100, help="Max training examples"
    )
    parser.add_argument(
        "--max-eval", type=int, default=50, help="Max evaluation examples"
    )
    parser.add_argument(
        "--num-candidates", type=int, default=10, help="MIPROv2 candidates"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )

    args = parser.parse_args()

    if args.all:
        optimize_all(api_base=args.api_base)
        return

    if not args.model or not args.specialty or not args.dataset:
        parser.error(
            "Provide --model, --specialty, --dataset, or use --all"
        )
        return

    # Resolve model ID
    model_info = MODEL_REGISTRY.get(args.model, {})
    actual_model = model_info.get("model_id", args.model)
    api_base = model_info.get("api_base", args.api_base)

    result = optimize_pair(
        model_id=actual_model,
        specialty=args.specialty,
        dataset_name=args.dataset,
        max_train=args.max_train,
        max_eval=args.max_eval,
        num_candidates=args.num_candidates,
        api_base=api_base,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    save_result(result, output_dir=output_dir)

    print(f"\n  Final result:")
    print(f"    Baseline:  {result.get('baseline_accuracy', 'N/A')}")
    print(f"    Optimized: {result.get('optimized_accuracy', 'N/A')}")
    print(f"    Delta:     {result.get('improvement', 'N/A')}")


if __name__ == "__main__":
    main()
