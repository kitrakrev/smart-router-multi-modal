#!/usr/bin/env python3
"""
Fine-tune a lightweight query router on RouterArena evaluation data.

Trains a linear classifier on top of MiniLM-L6-v2 (22M params) embeddings
to predict which model is the cheapest correct choice for a given query.

Labels:
  0 = gpt-4o-mini        (cheapest tier)
  1 = claude-3-haiku      (mid tier)
  2 = gemini-2.0-flash    (mid tier)

Usage:
  python finetune_router.py                   # train + eval
  python finetune_router.py --eval-only       # eval saved model
  python finetune_router.py --device cuda     # force GPU
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/Users/karthikraja/Desktop/TEST/a10/RouterArena/cached_results")
MODEL_DIR = Path(
    "/Users/karthikraja/Desktop/TEST/a10/router-prototype/models/routerarena-classifier"
)
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model files and their canonical names / cost tiers (lower = cheaper)
MODEL_FILES = {
    "gpt-4o-mini.jsonl": {"name": "gpt-4o-mini", "label": 0, "cost_rank": 0},
    "claude-3-haiku-20240307.jsonl": {"name": "claude-3-haiku", "label": 1, "cost_rank": 1},
    "gemini-2.0-flash-001.jsonl": {"name": "gemini-2.0-flash", "label": 2, "cost_rank": 1},
}

LABEL_NAMES = ["gpt-4o-mini", "claude-3-haiku", "gemini-2.0-flash"]
NUM_CLASSES = len(LABEL_NAMES)

# Training hyper-parameters
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 8  # early-stopping patience

SEED = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cached_results() -> dict[str, dict]:
    """Load all JSONL files. Returns {global_index: {model_name: row}}."""
    index_to_results: dict[str, dict] = defaultdict(dict)
    for filename, meta in MODEL_FILES.items():
        path = CACHE_DIR / filename
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                gidx = row["global_index"]
                score = row.get("evaluation_result", {}).get("score", 0.0)
                index_to_results[gidx][meta["name"]] = {
                    "question": row["question"],
                    "score": float(score),
                    "cost_rank": meta["cost_rank"],
                    "label": meta["label"],
                }
    return index_to_results


def build_dataset(index_to_results: dict) -> tuple[list, list, list, list]:
    """
    Build labelled dataset from queries that appear in ALL 3 model caches.

    Label = the cheapest model that got the query correct.
    If no model got it right, assign cheapest model (gpt-4o-mini) as fallback.

    Also returns gpt-only queries (those only in gpt-4o-mini) for generalization eval.
    """
    overlap_texts, overlap_labels = [], []
    gpt_only_texts, gpt_only_labels = [], []

    all_model_names = {m["name"] for m in MODEL_FILES.values()}

    for gidx, models in index_to_results.items():
        present_models = set(models.keys())

        # --- Overlap set: query exists in all 3 caches ---
        if present_models == all_model_names:
            question = next(iter(models.values()))["question"]

            # Find cheapest correct model
            correct = [
                (info["cost_rank"], info["label"])
                for info in models.values()
                if info["score"] >= 0.5
            ]
            if correct:
                correct.sort(key=lambda x: x[0])  # sort by cost
                label = correct[0][1]
            else:
                # No model correct -- default to cheapest
                label = 0

            overlap_texts.append(question)
            overlap_labels.append(label)

        # --- GPT-only set: only in gpt-4o-mini ---
        elif present_models == {"gpt-4o-mini"}:
            info = models["gpt-4o-mini"]
            question = info["question"]
            # For gpt-only, we can only check if gpt got it right
            # Label 0 if correct, still 0 if wrong (only model available)
            gpt_only_texts.append(question)
            gpt_only_labels.append(0)

    return overlap_texts, overlap_labels, gpt_only_texts, gpt_only_labels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class RouterClassifier(nn.Module):
    """Two-layer classifier head on frozen embeddings."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def encode_texts(
    encoder: SentenceTransformer, texts: list[str], batch_size: int = 256
) -> np.ndarray:
    """Encode list of texts into embeddings (numpy array)."""
    print(f"  Encoding {len(texts)} texts ...")
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    return embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> RouterClassifier:
    """Train classifier with early stopping on validation accuracy."""
    input_dim = X_train.shape[1]
    model = RouterClassifier(input_dim, NUM_CLASSES).to(device)

    # Convert to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_X = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Class weights to handle imbalance
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Acc':>8} | {'LR':>10}")
    print("-" * 48)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:6d} | {avg_loss:11.4f} | {val_acc:8.4f} | {lr_now:10.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_baseline(y: np.ndarray) -> float:
    """Baseline: always predict the most frequent class."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return counts.max() / len(y)


def compute_optimal_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Optimal selection rate: fraction of queries where the router
    picked a model that actually got the answer correct.
    Since labels encode 'cheapest correct model', matching the label
    means we picked the optimal route.
    """
    return (y_true == y_pred).mean()


def evaluate(
    model: RouterClassifier,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    split_name: str = "Test",
) -> dict:
    """Run evaluation and print results."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y, preds)
    baseline = compute_baseline(y)
    optimal_rate = compute_optimal_rate(y, preds)

    print(f"\n{'=' * 60}")
    print(f"  {split_name} Evaluation")
    print(f"{'=' * 60}")
    print(f"  Samples:            {len(y)}")
    print(f"  Baseline accuracy:  {baseline:.4f} (always predict majority class)")
    print(f"  Router accuracy:    {acc:.4f}")
    print(f"  Improvement:        {acc - baseline:+.4f} ({(acc - baseline) / baseline * 100:+.1f}%)")
    print(f"  Optimal selection:  {optimal_rate:.4f}")
    print()
    print("  Classification Report:")
    print(
        classification_report(
            y, preds, target_names=LABEL_NAMES, digits=4, zero_division=0
        )
    )
    print("  Confusion Matrix:")
    cm = confusion_matrix(y, preds, labels=list(range(NUM_CLASSES)))
    header = "  " + " " * 16 + "  ".join(f"{n:>14}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>14}" for v in row)
        print(f"  {LABEL_NAMES[i]:>14}  {row_str}")
    print()

    # Label distribution
    print("  Label distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = (y == i).sum()
        print(f"    {name}: {count} ({count / len(y) * 100:.1f}%)")

    return {"accuracy": acc, "baseline": baseline, "improvement": acc - baseline}


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model: RouterClassifier, metadata: dict):
    """Save classifier weights and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "classifier.pt")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Model saved to {MODEL_DIR}")


def load_model(device: torch.device, input_dim: int) -> RouterClassifier:
    """Load saved classifier."""
    model = RouterClassifier(input_dim, NUM_CLASSES)
    state = torch.load(MODEL_DIR / "classifier.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Loaded model from {MODEL_DIR}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune router classifier")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'mps', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate a previously saved model",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS})"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=ENCODER_NAME,
        help=f"Sentence-transformer encoder (default {ENCODER_NAME})",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading cached results ...")
    index_to_results = load_cached_results()
    overlap_texts, overlap_labels, gpt_only_texts, gpt_only_labels = build_dataset(
        index_to_results
    )
    overlap_labels = np.array(overlap_labels)
    gpt_only_labels = np.array(gpt_only_labels)

    print(f"  Overlap queries (all 3 models): {len(overlap_texts)}")
    print(f"  GPT-only queries:               {len(gpt_only_texts)}")

    # Label distribution in overlap set
    print("\n  Overlap label distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = (overlap_labels == i).sum()
        print(f"    {name}: {count} ({count / len(overlap_labels) * 100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Train/test split (80/20) on overlap set
    # ------------------------------------------------------------------
    print("\n[2/5] Splitting data (80/20) ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        overlap_texts,
        overlap_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=overlap_labels,
    )
    print(f"  Train: {len(X_train_text)}  |  Test: {len(X_test_text)}")

    # ------------------------------------------------------------------
    # 3. Encode with sentence-transformer
    # ------------------------------------------------------------------
    print(f"\n[3/5] Encoding with {args.encoder} ...")
    encoder = SentenceTransformer(args.encoder, device=str(device))
    emb_dim = encoder.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {emb_dim}")

    X_train_emb = encode_texts(encoder, X_train_text)
    X_test_emb = encode_texts(encoder, X_test_text)

    if gpt_only_texts:
        X_gpt_only_emb = encode_texts(encoder, gpt_only_texts)

    # ------------------------------------------------------------------
    # 4. Train or load classifier
    # ------------------------------------------------------------------
    global EPOCHS
    EPOCHS = args.epochs

    if args.eval_only:
        print("\n[4/5] Loading saved model ...")
        model = load_model(device, emb_dim)
    else:
        print(f"\n[4/5] Training classifier (epochs={EPOCHS}, bs={BATCH_SIZE}) ...")
        model = train_classifier(X_train_emb, y_train, X_test_emb, y_test, device)

        metadata = {
            "encoder": args.encoder,
            "embedding_dim": emb_dim,
            "num_classes": NUM_CLASSES,
            "label_names": LABEL_NAMES,
            "hidden_dim": 128,
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
            "overlap_size": len(overlap_texts),
            "epochs_config": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        }
        save_model(model, metadata)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluation ...")

    test_results = evaluate(model, X_test_emb, y_test, device, split_name="Overlap Test (20%)")

    if gpt_only_texts:
        gpt_results = evaluate(
            model,
            X_gpt_only_emb,
            gpt_only_labels,
            device,
            split_name="GPT-only Generalization",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Encoder:                {args.encoder}")
    print(f"  Classifier params:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Overlap test accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Majority baseline:      {test_results['baseline']:.4f}")
    print(f"  Improvement:            {test_results['improvement']:+.4f}")
    if gpt_only_texts:
        print(f"  GPT-only accuracy:      {gpt_results['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
