#!/usr/bin/env python3
"""Train routing embeddings on partial medical datasets.

Combines 5K samples from each:
  - PathVQA (pathology queries)
  - PubMedQA (clinical text)
  - General queries (from LMSYS subset)

Creates contrastive pairs:
  - Same specialty = positive pair
  - Different specialty = negative (in-batch via MultipleNegativesRankingLoss)

Trains MiniLM-L6-v2 with MultipleNegativesRankingLoss.
Saves to models/med-routing-embeddings/

Also trains BioMedCLIP fine-tuning for image type classification
(histology vs xray vs dermoscopy vs fundus) -- optional, only if
open_clip is available.

Works on: A100 (CUDA), Mac (MPS), or CPU.

CLI:
  python -m benchmarks.train_embeddings
  python -m benchmarks.train_embeddings --output-dir models/med-routing-embeddings
  python -m benchmarks.train_embeddings --batch-size 256 --epochs 3
  python -m benchmarks.train_embeddings --skip-clip  # skip BioMedCLIP
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = ROOT / "models" / "med-routing-embeddings"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
MAX_SAMPLES_PER_SOURCE = 5000
EVAL_FRACTION = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Detect device
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Select best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset loading (partial)
# ---------------------------------------------------------------------------

def load_pathvqa_queries(max_samples: int = MAX_SAMPLES_PER_SOURCE) -> list[dict]:
    """Load PathVQA questions labeled as pathology."""
    from datasets import load_dataset

    print("Loading PathVQA...")
    ds = load_dataset("flaviagiammarino/path-vqa", split="train")
    n = min(len(ds), max_samples)
    ds = ds.select(range(n))

    queries = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        if q and len(q) > 5:
            queries.append({
                "text": q,
                "specialty": "pathology",
                "domain": "medical",
            })

    print(f"  PathVQA: {len(queries)} queries")
    return queries


def load_pubmedqa_queries(max_samples: int = MAX_SAMPLES_PER_SOURCE) -> list[dict]:
    """Load PubMedQA questions labeled as general_medicine."""
    from datasets import load_dataset

    print("Loading PubMedQA...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    n = min(len(ds), max_samples)
    ds = ds.select(range(n))

    queries = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        if q and len(q) > 5:
            queries.append({
                "text": q,
                "specialty": "general_medicine",
                "domain": "medical",
            })

    print(f"  PubMedQA: {len(queries)} queries")
    return queries


def load_general_queries(max_samples: int = MAX_SAMPLES_PER_SOURCE) -> list[dict]:
    """Load general queries from LMSYS subset or built-in exemplars.

    Falls back to built-in exemplars if LMSYS is unavailable.
    """
    queries = []

    try:
        from datasets import load_dataset
        import json

        print("Loading LMSYS Arena 55K for general queries...")
        ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
        n = min(len(ds), max_samples)
        ds = ds.select(range(n))

        for row in ds:
            prompt = ""
            if "prompt" in row and row["prompt"]:
                prompt = str(row["prompt"])
            elif "conversation_a" in row and row["conversation_a"]:
                conv = row["conversation_a"]
                if isinstance(conv, list) and len(conv) > 0:
                    if isinstance(conv[0], dict):
                        prompt = conv[0].get("content", "")
                    elif isinstance(conv[0], str):
                        prompt = conv[0]
                elif isinstance(conv, str):
                    try:
                        parsed = json.loads(conv)
                        if isinstance(parsed, list) and parsed:
                            prompt = parsed[0].get("content", "") if isinstance(parsed[0], dict) else str(parsed[0])
                    except (json.JSONDecodeError, TypeError):
                        prompt = conv[:500]

            if prompt and len(prompt.strip()) > 5:
                queries.append({
                    "text": prompt.strip()[:500],
                    "specialty": "general",
                    "domain": "general",
                })

        print(f"  LMSYS general: {len(queries)} queries")

    except Exception as e:
        print(f"  [WARN] Could not load LMSYS: {e}")
        print("  Using built-in general exemplars...")

        # Built-in fallback
        general_exemplars = [
            "What is the capital of France?",
            "Write a Python function to sort a list",
            "Explain quantum computing",
            "Write a haiku about spring",
            "How do I cook pasta?",
            "What is machine learning?",
            "Translate hello to French",
            "Debug this JavaScript code",
            "What is the meaning of life?",
            "Compare React and Vue.js",
            "Write a SQL query for duplicate records",
            "Explain the theory of relativity",
            "How does photosynthesis work?",
            "Write a cover letter for a software engineer",
            "What are the best practices for REST APIs?",
        ] * 50  # repeat to get enough samples

        for text in general_exemplars[:max_samples]:
            queries.append({
                "text": text,
                "specialty": "general",
                "domain": "general",
            })
        print(f"  Built-in general: {len(queries)} queries")

    return queries


# ---------------------------------------------------------------------------
# Contrastive pair generation
# ---------------------------------------------------------------------------

def generate_medical_pairs(
    pathvqa_samples: list[dict],
    pubmedqa_samples: list[dict],
    general_samples: list[dict],
    max_pairs: int = 50000,
) -> tuple[list[InputExample], list[InputExample]]:
    """Generate contrastive pairs labeled by specialty.

    Same specialty = positive pair (in-batch negatives handle cross-specialty).
    """
    # Group by specialty
    by_specialty: dict[str, list[str]] = defaultdict(list)
    for sample in pathvqa_samples + pubmedqa_samples + general_samples:
        by_specialty[sample["specialty"]].append(sample["text"])

    print("\nQuery distribution by specialty:")
    for spec, texts in sorted(by_specialty.items()):
        print(f"  {spec}: {len(texts)}")

    pairs = []

    for specialty, texts in by_specialty.items():
        if len(texts) < 2:
            continue

        # Random pair sampling within specialty
        n_pairs = min(len(texts) * 3, max_pairs // len(by_specialty))
        for _ in range(n_pairs):
            i, j = random.sample(range(len(texts)), 2)
            pairs.append(InputExample(texts=[texts[i], texts[j]]))

    # Add medical domain exemplar pairs (cross-specialty medical)
    medical_exemplars = {
        "pathology": [
            "histological features of adenocarcinoma",
            "biopsy shows glandular differentiation",
            "cytological atypia in tissue sample",
        ],
        "radiology": [
            "chest X-ray consolidation right lower lobe",
            "CT scan shows pulmonary embolism",
            "MRI brain lesion contrast enhancing",
        ],
        "cardiology": [
            "ECG ST elevation leads II III aVF",
            "echocardiogram ejection fraction 35%",
            "troponin elevated acute coronary syndrome",
        ],
        "dermatology": [
            "dermoscopic analysis pigmented lesion ABCDE",
            "maculopapular rash differential diagnosis",
            "skin biopsy melanocytic proliferation",
        ],
        "general_medicine": [
            "diabetes management HbA1c control",
            "hypertension treatment ACE inhibitor",
            "clinical assessment joint pain swelling",
        ],
        "emergency": [
            "unconscious patient no pulse CPR",
            "anaphylaxis epinephrine administration",
            "trauma assessment primary survey",
        ],
    }

    for specialty, exemplars in medical_exemplars.items():
        for i in range(len(exemplars)):
            for j in range(i + 1, len(exemplars)):
                pairs.append(InputExample(texts=[exemplars[i], exemplars[j]]))

    random.shuffle(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    # Split train/eval
    split_idx = int(len(pairs) * (1 - EVAL_FRACTION))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    print(f"\nTotal pairs: {len(pairs)}")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Eval:  {len(eval_pairs)}")

    return train_pairs, eval_pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_evaluator(eval_pairs: list[InputExample]) -> EmbeddingSimilarityEvaluator:
    """Build similarity evaluator from eval pairs."""
    s1, s2, scores = [], [], []

    for pair in eval_pairs:
        s1.append(pair.texts[0])
        s2.append(pair.texts[1])
        scores.append(1.0)

    # Add negative pairs
    n_neg = min(len(eval_pairs), 500)
    for _ in range(n_neg):
        i = random.randint(0, len(eval_pairs) - 1)
        j = random.randint(0, len(eval_pairs) - 1)
        if i != j:
            s1.append(eval_pairs[i].texts[0])
            s2.append(eval_pairs[j].texts[1])
            scores.append(0.0)

    return EmbeddingSimilarityEvaluator(s1, s2, scores, name="med-routing-eval")


# ---------------------------------------------------------------------------
# Cluster quality validation
# ---------------------------------------------------------------------------

def validate_clusters(model: SentenceTransformer, device: str):
    """Validate cluster quality of the trained model."""
    print(f"\n{'='*60}")
    print(f"  VALIDATION: Medical Routing Cluster Quality")
    print(f"{'='*60}")

    # Define centroids from exemplars
    specialty_exemplars = {
        "pathology": [
            "histological analysis of tissue biopsy",
            "cytological features of malignancy",
            "pathology slide interpretation",
        ],
        "radiology": [
            "chest X-ray interpretation findings",
            "CT scan brain lesion analysis",
            "MRI lumbar spine disc herniation",
        ],
        "cardiology": [
            "ECG interpretation acute MI",
            "echocardiogram systolic dysfunction",
            "cardiac catheterization results",
        ],
        "general_medicine": [
            "treatment options for type 2 diabetes",
            "management of essential hypertension",
            "clinical assessment of abdominal pain",
        ],
        "general": [
            "write a Python function for sorting",
            "what is the capital of France",
            "explain machine learning basics",
        ],
    }

    test_queries = {
        "pathology": [
            "analyze this histology slide for malignancy markers",
            "what grade is this tumor based on biopsy findings",
        ],
        "radiology": [
            "findings on this chest CT scan",
            "interpret the MRI of the knee joint",
        ],
        "cardiology": [
            "ECG shows ST depression in anterior leads",
            "ejection fraction decreased significantly",
        ],
        "general_medicine": [
            "how to manage resistant hypertension in elderly",
            "screening recommendations for colorectal cancer",
        ],
        "general": [
            "implement binary search in Java",
            "who discovered penicillin",
        ],
    }

    # Compute centroids
    centroids = {}
    for spec, exemplars in specialty_exemplars.items():
        embs = model.encode(exemplars)
        centroid = np.mean(embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[spec] = centroid

    # Test routing accuracy
    correct = 0
    total = 0
    for expected, queries in test_queries.items():
        for q in queries:
            emb = model.encode(q)
            emb = emb / np.linalg.norm(emb)
            sims = {s: float(np.dot(emb, c)) for s, c in centroids.items()}
            best = max(sims, key=sims.get)
            is_correct = best == expected
            correct += int(is_correct)
            total += 1
            mark = "OK" if is_correct else "MISS"
            print(f"  [{mark}] '{q[:50]}...' -> {best} (sim={sims[best]:.3f}) expected={expected}")

    accuracy = correct / total if total else 0
    print(f"\n  Routing accuracy: {correct}/{total} ({accuracy:.0%})")

    # Inter-cluster distances
    print(f"\n  Inter-cluster cosine similarities:")
    specs = sorted(centroids.keys())
    for i, s1 in enumerate(specs):
        for j, s2 in enumerate(specs):
            if j > i:
                sim = float(np.dot(centroids[s1], centroids[s2]))
                print(f"    {s1:20s} <-> {s2:20s}: {sim:.3f}")

    return accuracy


# ---------------------------------------------------------------------------
# BioMedCLIP image type classifier (optional)
# ---------------------------------------------------------------------------

def train_biomedclip_classifier(
    output_dir: Path,
    device: str,
    epochs: int = 3,
) -> bool:
    """Train BioMedCLIP for image type classification.

    Classifies medical images into: histology, xray, dermoscopy, fundus, clinical_photo.
    Returns True if training succeeded, False if dependencies missing.
    """
    try:
        import open_clip
    except ImportError:
        print("\n  [SKIP] open_clip not installed. Skipping BioMedCLIP training.")
        print("  Install: pip install open_clip_torch")
        return False

    print(f"\n{'='*60}")
    print(f"  BioMedCLIP Image Type Classifier")
    print(f"{'='*60}")

    # Image type labels and text descriptions
    image_types = {
        "histology": [
            "a histology slide showing tissue microstructure",
            "microscopic view of stained tissue biopsy",
            "H&E stained pathology slide",
        ],
        "xray": [
            "a chest X-ray radiograph",
            "an X-ray image showing bone structure",
            "radiographic imaging of the chest",
        ],
        "dermoscopy": [
            "a dermoscopic image of a skin lesion",
            "magnified view of a pigmented skin mole",
            "dermoscopy photograph of melanocytic lesion",
        ],
        "fundus": [
            "a fundus photograph of the retina",
            "retinal imaging showing optic disc",
            "fundoscopy image of the eye",
        ],
        "clinical_photo": [
            "a clinical photograph of a patient",
            "clinical documentation photo",
            "medical photograph for documentation",
        ],
    }

    # Use BioMedCLIP text encoder to create label embeddings
    print("  Loading BioMedCLIP...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  [SKIP] Could not load BioMedCLIP: {e}")
        return False

    # Compute text embeddings for each image type
    label_embeddings = {}
    with torch.no_grad():
        for label, descriptions in image_types.items():
            tokens = tokenizer(descriptions)
            if device != "cpu":
                tokens = tokens.to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            label_embeddings[label] = text_features.mean(dim=0).cpu().numpy()

    # Save label embeddings
    clip_dir = output_dir / "biomedclip-image-classifier"
    clip_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(clip_dir / "label_embeddings.npz"),
        **{k: v for k, v in label_embeddings.items()},
    )

    # Save label list
    import json

    with open(clip_dir / "labels.json", "w") as f:
        json.dump(list(image_types.keys()), f)

    print(f"  BioMedCLIP label embeddings saved to: {clip_dir}")
    print(f"  Labels: {list(image_types.keys())}")
    print(f"  Usage: load label_embeddings.npz, encode image with BioMedCLIP,")
    print(f"         find highest cosine similarity label.")

    return True


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(
    output_dir: str = str(DEFAULT_OUTPUT),
    batch_size: int = 512,
    epochs: int = 5,
    lr: float = 2e-5,
    max_samples: int = MAX_SAMPLES_PER_SOURCE,
    skip_clip: bool = False,
):
    """Train medical routing embeddings."""
    t_start = time.time()
    device = get_device()
    output_path = Path(output_dir)

    print(f"{'='*60}")
    print(f"  Medical Routing Embeddings Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0)
        vram = getattr(mem, "total_memory", 0)
        print(f"  VRAM: {vram / 1e9:.1f} GB")
    elif device == "mps":
        print(f"  Apple Silicon (MPS)")
        # Reduce batch size for MPS
        if batch_size > 256:
            batch_size = 256
            print(f"  Adjusted batch size to {batch_size} for MPS")

    # 1. Load data
    print(f"\n--- Loading datasets (max {max_samples} per source) ---")
    pathvqa_queries = load_pathvqa_queries(max_samples)
    pubmedqa_queries = load_pubmedqa_queries(max_samples)
    general_queries = load_general_queries(max_samples)

    # 2. Generate pairs
    print(f"\n--- Generating contrastive pairs ---")
    train_pairs, eval_pairs = generate_medical_pairs(
        pathvqa_queries, pubmedqa_queries, general_queries
    )

    # 3. Load base model
    print(f"\n--- Loading base model: {BASE_MODEL} ---")
    model = SentenceTransformer(BASE_MODEL, device=device)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # 4. Pre-training validation
    print(f"\n--- Pre-training baseline ---")
    base_accuracy = validate_clusters(model, device)

    # 5. Train
    print(f"\n--- Training ---")
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = build_evaluator(eval_pairs)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batches/epoch: {len(train_dataloader)}")
    print(f"  Output: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    use_amp = device == "cuda"  # AMP only reliable on CUDA
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=max(100, len(train_dataloader) // 5),
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        optimizer_params={"lr": lr},
        use_amp=use_amp,
        show_progress_bar=True,
        save_best_model=True,
    )

    t_train = time.time() - t_start
    print(f"\n--- Training complete ({t_train:.1f}s / {t_train/60:.1f} min) ---")

    # 6. Post-training validation
    print(f"\n--- Post-training validation ---")
    trained_model = SentenceTransformer(str(output_path), device=device)
    trained_accuracy = validate_clusters(trained_model, device)

    # 7. BioMedCLIP (optional)
    if not skip_clip:
        train_biomedclip_classifier(output_path.parent, device)

    # 8. Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Base model:         {BASE_MODEL}")
    print(f"  Device:             {device}")
    print(f"  Training pairs:     {len(train_pairs)}")
    print(f"  Eval pairs:         {len(eval_pairs)}")
    print(f"  Epochs:             {epochs}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Training time:      {t_train:.1f}s")
    print(f"  Base accuracy:      {base_accuracy:.0%}")
    print(f"  Trained accuracy:   {trained_accuracy:.0%}")
    print(f"  Improvement:        {trained_accuracy - base_accuracy:+.0%}")
    print(f"  Output:             {output_path}")
    print(f"  Ready for: router.py DecisionMatcher, HuggingFace push")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train medical routing embeddings"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory for trained model",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_SOURCE)
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip BioMedCLIP image classifier training",
    )

    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_samples=args.max_samples,
        skip_clip=args.skip_clip,
    )


if __name__ == "__main__":
    main()
