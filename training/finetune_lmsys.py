#!/usr/bin/env python3
"""
Fine-tune a multi-head task-profile classifier on LMSYS 55K data.

KEY PRINCIPLE: No model names in training labels.
Instead of "query -> pick gpt-4o", we train "query -> task profile" where
the task profile describes WHAT the query needs (complexity, reasoning, tools, etc).

Model selection happens at inference time by mapping task profiles to models
via config rules (cost/capability matching).

Multi-head outputs:
  - task_type:     7 classes (QA, code, math, creative, translation, reasoning, chat)
  - complexity:    float [0,1]
  - needs_reasoning: float [0,1] (binary sigmoid)
  - needs_vision:  float [0,1] (binary sigmoid)
  - needs_tools:   float [0,1] (binary sigmoid)
  - temperature:   float [0,1]
  - max_tokens:    float -> scaled to [128, 4096]
  - thinking_tokens: float -> scaled to [0, 2000]
  - cost_sensitivity: 3 classes (low/medium/high)

Training data: LMSYS Arena Human Preference 55K
Labels: derived from query text features (rule-based), NOT from model names.

Usage:
  python finetune_lmsys.py                    # train + eval
  python finetune_lmsys.py --device cuda      # force GPU
  python finetune_lmsys.py --eval-only        # eval saved model
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent.parent / "models" / "lmsys-task-classifier"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Task type labels
TASK_TYPES = ["qa", "code", "math", "creative", "translation", "reasoning", "chat"]
TASK_TYPE_TO_IDX = {t: i for i, t in enumerate(TASK_TYPES)}
NUM_TASK_TYPES = len(TASK_TYPES)

# Cost sensitivity labels
COST_LABELS = ["low", "medium", "high"]
COST_TO_IDX = {c: i for i, c in enumerate(COST_LABELS)}
NUM_COST_CLASSES = len(COST_LABELS)

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 6
SEED = 42

# ---------------------------------------------------------------------------
# Rule-based label derivation (NO model names used)
# ---------------------------------------------------------------------------

# Keyword patterns for task type classification
_CODE_PATTERNS = [
    r'\b(write|implement|create|build|code|debug|fix|refactor)\b.*\b(function|class|program|script|code|app|api|server|module)\b',
    r'\b(python|javascript|java|c\+\+|rust|go|typescript|html|css|sql|bash|ruby|swift|kotlin)\b',
    r'\b(import|def |class |function |const |let |var |public |private )\b',
    r'\b(bug|error|exception|traceback|stack\s*trace|compile|runtime)\b',
    r'\b(algorithm|data\s*structure|binary\s*search|sort|hash|tree|graph|linked\s*list)\b',
    r'```',  # code blocks
]

_MATH_PATTERNS = [
    r'\b(solve|prove|derive|calculate|compute|evaluate|simplify|factor)\b',
    r'\b(equation|integral|derivative|matrix|vector|eigenvalue|polynomial)\b',
    r'\b(theorem|lemma|corollary|proof|qed|contradiction)\b',
    r'\b(algebra|calculus|geometry|trigonometry|statistics|probability)\b',
    r'\b(sin|cos|tan|log|ln|exp|sqrt|lim|sum|product)\b[\s(]',
    r'[=+\-*/^]\s*\d+.*[=+\-*/^]',  # math expressions
    r'\b\d+\s*[+\-*/^]\s*\d+\b',
]

_CREATIVE_PATTERNS = [
    r'\b(write|compose|create)\s+(a\s+)?(poem|story|essay|song|haiku|limerick|novel|screenplay|dialogue)\b',
    r'\b(creative|fiction|imagine|fantasy|narrative|literary)\b',
    r'\b(character|plot|setting|theme|metaphor|allegory|protagonist)\b',
    r'\b(rhyme|verse|stanza|sonnet|ballad)\b',
]

_TRANSLATION_PATTERNS = [
    r'\b(translate|translation|translating)\b',
    r'\b(english|french|spanish|german|chinese|japanese|korean|arabic|hindi|portuguese|russian|italian)\b.*\b(to|into|from)\b.*\b(english|french|spanish|german|chinese|japanese|korean|arabic|hindi|portuguese|russian|italian)\b',
    r'\b(en|fr|es|de|zh|ja|ko|ar|hi|pt|ru|it)\s*[-=:>]+\s*(en|fr|es|de|zh|ja|ko|ar|hi|pt|ru|it)\b',
]

_REASONING_PATTERNS = [
    r'\b(explain|why|reason|logic|deduce|infer|analyze|evaluate|compare|contrast)\b.*\b(because|since|therefore|thus|hence)\b',
    r'\b(step\s*by\s*step|chain\s*of\s*thought|think\s*through|let\'?s\s*think|reasoning)\b',
    r'\b(pros?\s*and\s*cons?|trade-?off|advantage|disadvantage|benefit|drawback)\b',
    r'\b(if\s+.*then|assuming|given\s+that|suppose|consider)\b',
    r'\b(argument|conclusion|premise|hypothesis|evidence|claim)\b',
]

_QA_PATTERNS = [
    r'\b(what|who|when|where|which|how\s+many|how\s+much|how\s+long|how\s+old)\b.*\?',
    r'\b(define|definition|meaning|explain)\b.*\b(of|is|are|was|were)\b',
    r'\b(fact|true|false|correct|incorrect|answer)\b',
    r'\b(capital|population|president|founder|inventor|discovery)\b',
    r'\b(list|name|identify|describe)\b.*\b(the|all|main|key|important)\b',
]

# Technical terms for complexity estimation
_TECHNICAL_TERMS = {
    "algorithm", "optimization", "architecture", "distributed", "concurrent",
    "asymptotic", "polynomial", "differential", "stochastic", "recurrence",
    "eigenvalue", "topology", "morphism", "isomorphism", "homogeneous",
    "heterogeneous", "bayesian", "regression", "gradient", "backpropagation",
    "integral", "derivative", "theorem", "epsilon", "convergence",
    "divergence", "fourier", "laplace", "kubernetes", "microservice",
    "terraform", "cryptography", "recursion", "heuristic", "deterministic",
    "nondeterministic", "quantum", "neural", "transformer", "attention",
    "embedding", "latent", "manifold", "convex", "nonconvex", "entropy",
    "markov", "hamiltonian", "lagrangian", "tensor", "jacobian", "hessian",
}

# Tool-related keywords
_TOOL_KEYWORDS = [
    "search", "look up", "find online", "browse", "web", "url", "http",
    "api call", "fetch", "download", "upload", "database", "query the",
    "run this", "execute", "terminal", "command line", "pip install",
    "google", "wikipedia", "stack overflow",
]

# Vision keywords
_VISION_KEYWORDS = [
    "image", "picture", "photo", "screenshot", "diagram", "chart",
    "graph", "plot", "figure", "visual", "see this", "look at this",
    "attached", "shown above", "displayed", "png", "jpg", "jpeg",
]


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in text."""
    count = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return count


def estimate_complexity(text: str) -> float:
    """Estimate query complexity from text features. Returns [0, 1]."""
    words = text.lower().split()
    word_count = len(words)

    # Length factor
    length_score = min(word_count / 200, 1.0)

    # Technical term density
    tech_count = sum(1 for w in words if w.strip("?.,!:;()[]{}") in _TECHNICAL_TERMS)
    tech_score = min(tech_count / 4, 1.0)

    # Multi-step indicators
    sentence_count = max(text.count(".") + text.count("?") + text.count("!"), 1)
    multi_step = min(sentence_count / 5, 1.0)

    # Nested structure (code blocks, lists, indentation)
    has_code_block = 1.0 if "```" in text else 0.0
    has_numbered_list = 1.0 if re.search(r'\b\d+[.)]\s', text) else 0.0

    # Question depth
    question_words = {"what", "why", "how", "explain", "describe", "compare",
                      "analyze", "evaluate", "derive", "prove", "justify"}
    q_count = sum(1 for w in words if w.strip("?.,!") in question_words)
    q_score = min(q_count / 3, 1.0)

    score = (0.15 * length_score +
             0.30 * tech_score +
             0.15 * multi_step +
             0.10 * has_code_block +
             0.05 * has_numbered_list +
             0.25 * q_score)

    return round(min(max(score, 0.0), 1.0), 4)


def derive_task_profile(prompt_text: str) -> dict:
    """
    Derive a complete task profile from query text.
    NO model information is used -- purely text-based analysis.
    """
    text_lower = prompt_text.lower()
    words = text_lower.split()

    # --- Task type classification (priority-ordered) ---
    scores = {
        "code": _count_pattern_matches(prompt_text, _CODE_PATTERNS),
        "math": _count_pattern_matches(prompt_text, _MATH_PATTERNS),
        "creative": _count_pattern_matches(prompt_text, _CREATIVE_PATTERNS),
        "translation": _count_pattern_matches(prompt_text, _TRANSLATION_PATTERNS),
        "reasoning": _count_pattern_matches(prompt_text, _REASONING_PATTERNS),
        "qa": _count_pattern_matches(prompt_text, _QA_PATTERNS),
    }

    # Best matching type; default to "chat" if no strong signal
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        task_type = "chat"
    elif scores[best_type] <= 1 and len(words) < 10:
        # Very short with weak signal -> chat
        task_type = "chat"
    else:
        task_type = best_type

    # --- Complexity ---
    complexity = estimate_complexity(prompt_text)

    # --- Needs reasoning ---
    needs_reasoning = False
    if task_type == "math":
        needs_reasoning = True
    elif task_type == "reasoning":
        needs_reasoning = True
    elif task_type == "code" and complexity > 0.5:
        needs_reasoning = True
    elif complexity > 0.7:
        needs_reasoning = True

    # --- Needs vision ---
    vision_hits = sum(1 for kw in _VISION_KEYWORDS if kw in text_lower)
    needs_vision = vision_hits >= 2  # need at least 2 vision keywords

    # --- Needs tools ---
    tool_hits = sum(1 for kw in _TOOL_KEYWORDS if kw in text_lower)
    needs_tools = tool_hits >= 2

    # --- Recommended temperature ---
    if task_type == "math":
        temperature = 0.0
    elif task_type == "code":
        temperature = 0.2
    elif task_type == "translation":
        temperature = 0.3
    elif task_type in ("qa", "reasoning"):
        temperature = 0.4
    elif task_type == "creative":
        temperature = 0.9
    elif task_type == "chat":
        temperature = 0.7
    else:
        temperature = 0.5

    # --- Recommended max_tokens ---
    if task_type == "creative":
        max_tokens = 2048
    elif task_type == "code":
        max_tokens = 2048 if complexity > 0.5 else 1024
    elif task_type == "math":
        max_tokens = 1024 if complexity > 0.5 else 512
    elif task_type == "reasoning":
        max_tokens = 1536
    elif task_type == "translation":
        max_tokens = max(256, min(len(words) * 4, 2048))
    elif task_type == "qa":
        max_tokens = 512 if complexity < 0.3 else 1024
    elif task_type == "chat":
        max_tokens = 512 if complexity < 0.3 else 1024
    else:
        max_tokens = 1024

    # --- Estimated thinking tokens ---
    if not needs_reasoning:
        thinking_tokens = 0
    elif task_type == "math":
        thinking_tokens = int(500 + complexity * 1000)
    elif task_type == "reasoning":
        thinking_tokens = int(300 + complexity * 700)
    elif task_type == "code" and complexity > 0.5:
        thinking_tokens = int(200 + complexity * 500)
    else:
        thinking_tokens = int(100 + complexity * 400)

    # --- Cost sensitivity ---
    if complexity > 0.7 or needs_reasoning:
        cost_sensitivity = "low"  # complex tasks justify higher cost
    elif complexity < 0.3 and task_type in ("chat", "qa"):
        cost_sensitivity = "high"  # simple queries should be cheap
    else:
        cost_sensitivity = "medium"

    return {
        "task_type": task_type,
        "complexity": complexity,
        "needs_reasoning": needs_reasoning,
        "needs_vision": needs_vision,
        "needs_tools": needs_tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking_tokens": thinking_tokens,
        "cost_sensitivity": cost_sensitivity,
    }


def profile_to_tensors(profile: dict) -> dict:
    """Convert a task profile dict into tensor-ready values."""
    return {
        "task_type": TASK_TYPE_TO_IDX[profile["task_type"]],
        "complexity": profile["complexity"],
        "needs_reasoning": 1.0 if profile["needs_reasoning"] else 0.0,
        "needs_vision": 1.0 if profile["needs_vision"] else 0.0,
        "needs_tools": 1.0 if profile["needs_tools"] else 0.0,
        "temperature": profile["temperature"],
        "max_tokens": profile["max_tokens"] / 4096.0,  # normalize to [0,1]
        "thinking_tokens": profile["thinking_tokens"] / 2000.0,  # normalize to [0,1]
        "cost_sensitivity": COST_TO_IDX[profile["cost_sensitivity"]],
    }


# ---------------------------------------------------------------------------
# Multi-head classifier
# ---------------------------------------------------------------------------

class TaskProfileClassifier(nn.Module):
    """
    Multi-head classifier: 384-dim embedding -> task profile vector.

    Heads:
      - task_type: 7-class softmax
      - complexity: sigmoid -> [0,1]
      - needs_reasoning: sigmoid -> [0,1]
      - needs_vision: sigmoid -> [0,1]
      - needs_tools: sigmoid -> [0,1]
      - temperature: sigmoid -> [0,1]
      - max_tokens: sigmoid -> [0,1] (rescale to [128, 4096] at inference)
      - thinking_tokens: relu -> [0, inf] (rescale to [0, 2000] at inference)
      - cost_sensitivity: 3-class softmax
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Classification heads
        self.task_type_head = nn.Linear(hidden_dim, NUM_TASK_TYPES)
        self.cost_sensitivity_head = nn.Linear(hidden_dim, NUM_COST_CLASSES)

        # Regression heads (single outputs)
        self.complexity_head = nn.Linear(hidden_dim, 1)
        self.needs_reasoning_head = nn.Linear(hidden_dim, 1)
        self.needs_vision_head = nn.Linear(hidden_dim, 1)
        self.needs_tools_head = nn.Linear(hidden_dim, 1)
        self.temperature_head = nn.Linear(hidden_dim, 1)
        self.max_tokens_head = nn.Linear(hidden_dim, 1)
        self.thinking_tokens_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "task_type": self.task_type_head(h),               # (B, 7) logits
            "cost_sensitivity": self.cost_sensitivity_head(h), # (B, 3) logits
            "complexity": torch.sigmoid(self.complexity_head(h)).squeeze(-1),          # (B,)
            "needs_reasoning": torch.sigmoid(self.needs_reasoning_head(h)).squeeze(-1),# (B,)
            "needs_vision": torch.sigmoid(self.needs_vision_head(h)).squeeze(-1),      # (B,)
            "needs_tools": torch.sigmoid(self.needs_tools_head(h)).squeeze(-1),        # (B,)
            "temperature": torch.sigmoid(self.temperature_head(h)).squeeze(-1),        # (B,)
            "max_tokens": torch.sigmoid(self.max_tokens_head(h)).squeeze(-1),          # (B,)
            "thinking_tokens": F.relu(self.thinking_tokens_head(h)).squeeze(-1),       # (B,)
        }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(outputs: dict, targets: dict, device: torch.device) -> torch.Tensor:
    """Combined multi-head loss with task-appropriate weighting."""
    losses = {}

    # Classification losses (cross entropy)
    losses["task_type"] = F.cross_entropy(
        outputs["task_type"], targets["task_type"].to(device)
    )
    losses["cost_sensitivity"] = F.cross_entropy(
        outputs["cost_sensitivity"], targets["cost_sensitivity"].to(device)
    )

    # Regression losses (MSE for continuous, BCE for binary)
    losses["complexity"] = F.mse_loss(
        outputs["complexity"], targets["complexity"].to(device)
    )
    losses["needs_reasoning"] = F.binary_cross_entropy(
        outputs["needs_reasoning"], targets["needs_reasoning"].to(device)
    )
    losses["needs_vision"] = F.binary_cross_entropy(
        outputs["needs_vision"], targets["needs_vision"].to(device)
    )
    losses["needs_tools"] = F.binary_cross_entropy(
        outputs["needs_tools"], targets["needs_tools"].to(device)
    )
    losses["temperature"] = F.mse_loss(
        outputs["temperature"], targets["temperature"].to(device)
    )
    losses["max_tokens"] = F.mse_loss(
        outputs["max_tokens"], targets["max_tokens"].to(device)
    )
    losses["thinking_tokens"] = F.mse_loss(
        outputs["thinking_tokens"], targets["thinking_tokens"].to(device)
    )

    # Weighted combination (task_type and reasoning are most important)
    weights = {
        "task_type": 3.0,
        "complexity": 1.0,
        "needs_reasoning": 2.0,
        "needs_vision": 1.0,
        "needs_tools": 1.0,
        "temperature": 0.5,
        "max_tokens": 0.5,
        "thinking_tokens": 1.0,
        "cost_sensitivity": 2.0,
    }

    total = sum(weights[k] * losses[k] for k in losses)
    return total


# ---------------------------------------------------------------------------
# Data loading from LMSYS 55K
# ---------------------------------------------------------------------------

def extract_prompt_text(conversation) -> str:
    """Extract the user's prompt from the conversation field.

    The LMSYS dataset stores conversations as a list of [role, content] pairs
    or sometimes as a raw string. We extract only user messages.
    """
    if isinstance(conversation, str):
        return conversation

    if isinstance(conversation, list):
        parts = []
        for turn in conversation:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                role, content = turn[0], turn[1]
                if role in ("user", "human"):
                    parts.append(str(content))
            elif isinstance(turn, dict):
                if turn.get("role") in ("user", "human"):
                    parts.append(str(turn.get("content", "")))
        if parts:
            return " ".join(parts)

    return str(conversation)


def load_lmsys_data(max_samples: int = 0) -> list[str]:
    """Load LMSYS 55K dataset and extract prompts."""
    print("  Loading LMSYS Arena Human Preference 55K dataset...")
    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
    print(f"  Raw dataset size: {len(ds)}")

    prompts = []
    for row in ds:
        # The dataset has conversation_a and conversation_b columns
        # Both contain the same user prompt with different model responses
        conv = row.get("conversation_a") or row.get("conversation_b") or row.get("prompt", "")
        text = extract_prompt_text(conv)
        if text and len(text.strip()) > 5:
            prompts.append(text.strip())

    if max_samples > 0:
        prompts = prompts[:max_samples]

    print(f"  Extracted {len(prompts)} valid prompts")
    return prompts


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_texts(encoder: SentenceTransformer, texts: list[str],
                 batch_size: int = 512) -> np.ndarray:
    """Encode texts into embeddings."""
    print(f"  Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
    return embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def prepare_targets(profiles: list[dict]) -> dict[str, torch.Tensor]:
    """Convert list of profile dicts into target tensors."""
    tensor_profiles = [profile_to_tensors(p) for p in profiles]

    return {
        "task_type": torch.tensor([t["task_type"] for t in tensor_profiles], dtype=torch.long),
        "complexity": torch.tensor([t["complexity"] for t in tensor_profiles], dtype=torch.float32),
        "needs_reasoning": torch.tensor([t["needs_reasoning"] for t in tensor_profiles], dtype=torch.float32),
        "needs_vision": torch.tensor([t["needs_vision"] for t in tensor_profiles], dtype=torch.float32),
        "needs_tools": torch.tensor([t["needs_tools"] for t in tensor_profiles], dtype=torch.float32),
        "temperature": torch.tensor([t["temperature"] for t in tensor_profiles], dtype=torch.float32),
        "max_tokens": torch.tensor([t["max_tokens"] for t in tensor_profiles], dtype=torch.float32),
        "thinking_tokens": torch.tensor([t["thinking_tokens"] for t in tensor_profiles], dtype=torch.float32),
        "cost_sensitivity": torch.tensor([t["cost_sensitivity"] for t in tensor_profiles], dtype=torch.long),
    }


def train_model(
    X_train: np.ndarray, targets_train: dict,
    X_val: np.ndarray, targets_val: dict,
    device: torch.device,
    epochs: int = EPOCHS,
) -> TaskProfileClassifier:
    """Train multi-head classifier with early stopping."""
    input_dim = X_train.shape[1]
    model = TaskProfileClassifier(input_dim=input_dim).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Prepare training data
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(
        X_train_t,
        targets_train["task_type"],
        targets_train["complexity"],
        targets_train["needs_reasoning"],
        targets_train["needs_vision"],
        targets_train["needs_tools"],
        targets_train["temperature"],
        targets_train["max_tokens"],
        targets_train["thinking_tokens"],
        targets_train["cost_sensitivity"],
    )
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # Validation data on device
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_targets = {k: v.to(device) for k, v in targets_val.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'TT Acc':>7} | {'CS Acc':>7} | {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            xb = batch[0].to(device)
            batch_targets = {
                "task_type": batch[1].to(device),
                "complexity": batch[2].to(device),
                "needs_reasoning": batch[3].to(device),
                "needs_vision": batch[4].to(device),
                "needs_tools": batch[5].to(device),
                "temperature": batch[6].to(device),
                "max_tokens": batch[7].to(device),
                "thinking_tokens": batch[8].to(device),
                "cost_sensitivity": batch[9].to(device),
            }

            optimizer.zero_grad()
            outputs = model(xb)
            loss = compute_loss(outputs, batch_targets, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()
        avg_train_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = compute_loss(val_outputs, val_targets, device).item()

            # Task type accuracy
            tt_preds = val_outputs["task_type"].argmax(dim=1)
            tt_acc = (tt_preds == val_targets["task_type"]).float().mean().item()

            # Cost sensitivity accuracy
            cs_preds = val_outputs["cost_sensitivity"].argmax(dim=1)
            cs_acc = (cs_preds == val_targets["cost_sensitivity"]).float().mean().item()

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:6d} | {avg_train_loss:11.4f} | {val_loss:10.4f} | {tt_acc:7.4f} | {cs_acc:7.4f} | {lr_now:10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n  Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: TaskProfileClassifier, X: np.ndarray, targets: dict,
                   profiles: list[dict], device: torch.device, split_name: str = "Test"):
    """Comprehensive evaluation of multi-head model."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_t)

    results = {}

    # --- Task type accuracy ---
    tt_preds = outputs["task_type"].argmax(dim=1).cpu().numpy()
    tt_true = targets["task_type"].numpy()
    tt_acc = (tt_preds == tt_true).mean()
    results["task_type_accuracy"] = tt_acc

    # Per-class accuracy
    print(f"\n{'=' * 70}")
    print(f"  {split_name} Evaluation")
    print(f"{'=' * 70}")
    print(f"\n  Task Type Accuracy: {tt_acc:.4f}")
    print(f"  {'Type':<15} {'Count':>7} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*42}")
    for i, tname in enumerate(TASK_TYPES):
        mask = tt_true == i
        count = mask.sum()
        if count > 0:
            correct = (tt_preds[mask] == i).sum()
            acc = correct / count
            print(f"  {tname:<15} {count:>7} {correct:>8} {acc:>9.4f}")
        else:
            print(f"  {tname:<15} {count:>7} {'n/a':>8} {'n/a':>9}")

    # --- Cost sensitivity accuracy ---
    cs_preds = outputs["cost_sensitivity"].argmax(dim=1).cpu().numpy()
    cs_true = targets["cost_sensitivity"].numpy()
    cs_acc = (cs_preds == cs_true).mean()
    results["cost_sensitivity_accuracy"] = cs_acc
    print(f"\n  Cost Sensitivity Accuracy: {cs_acc:.4f}")
    for i, cname in enumerate(COST_LABELS):
        mask = cs_true == i
        count = mask.sum()
        if count > 0:
            correct = (cs_preds[mask] == i).sum()
            acc = correct / count
            print(f"  {cname:<15} {count:>7} {correct:>8} {acc:>9.4f}")

    # --- Binary classification metrics ---
    for bname in ["needs_reasoning", "needs_vision", "needs_tools"]:
        pred_binary = (outputs[bname].cpu().numpy() > 0.5).astype(int)
        true_binary = targets[bname].numpy().astype(int)
        acc = (pred_binary == true_binary).mean()
        results[f"{bname}_accuracy"] = acc
        pos_count = true_binary.sum()
        neg_count = len(true_binary) - pos_count
        print(f"\n  {bname} Accuracy: {acc:.4f} (pos={pos_count}, neg={neg_count})")

    # --- Regression metrics ---
    for rname in ["complexity", "temperature", "max_tokens", "thinking_tokens"]:
        pred = outputs[rname].cpu().numpy()
        true = targets[rname].numpy()
        mae = np.abs(pred - true).mean()
        results[f"{rname}_mae"] = mae
        print(f"  {rname} MAE: {mae:.4f}")

    return results


def evaluate_on_routerarena(model: TaskProfileClassifier, encoder: SentenceTransformer,
                             device: torch.device):
    """
    Test on RouterArena 8,400 queries:
    - Predict task profile for each query
    - Map profile -> model selection using config rules
    - Report distribution and simulated cost/latency
    """
    cache_dir = Path(__file__).parent.parent / "RouterArena" / "cached_results"
    if not cache_dir.exists():
        print(f"\n  [SKIP] RouterArena cache not found at {cache_dir}")
        return

    # Load queries from gpt-4o-mini.jsonl (largest set)
    gpt_file = cache_dir / "gpt-4o-mini.jsonl"
    if not gpt_file.exists():
        print(f"\n  [SKIP] {gpt_file} not found")
        return

    queries = []
    with open(gpt_file) as f:
        for line in f:
            row = json.loads(line)
            queries.append(row["question"])

    print(f"\n{'=' * 70}")
    print(f"  RouterArena Evaluation ({len(queries)} queries)")
    print(f"{'=' * 70}")

    # Encode all queries
    embeddings = encode_texts(encoder, queries, batch_size=512)

    # Predict task profiles
    model.eval()
    all_profiles = []
    batch_size = 1024
    for i in range(0, len(embeddings), batch_size):
        batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(batch)
        for j in range(batch.size(0)):
            profile = {
                "task_type": TASK_TYPES[outputs["task_type"][j].argmax().item()],
                "complexity": outputs["complexity"][j].item(),
                "needs_reasoning": outputs["needs_reasoning"][j].item() > 0.5,
                "needs_vision": outputs["needs_vision"][j].item() > 0.5,
                "needs_tools": outputs["needs_tools"][j].item() > 0.5,
                "temperature": outputs["temperature"][j].item(),
                "max_tokens": int(outputs["max_tokens"][j].item() * 4096),
                "thinking_tokens": int(outputs["thinking_tokens"][j].item() * 2000),
                "cost_sensitivity": COST_LABELS[outputs["cost_sensitivity"][j].argmax().item()],
            }
            all_profiles.append(profile)

    # --- Model cost table (from config.yaml) ---
    model_costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "latency": 800, "quality": 0.92},
        "claude-sonnet": {"input": 3.00, "output": 15.00, "latency": 600, "quality": 0.95},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "latency": 300, "quality": 0.70},
        "qwen-3b-local": {"input": 0.001, "output": 0.003, "latency": 50, "quality": 0.45},
    }

    # --- Map profiles to model selections ---
    model_selections = []
    total_cost = 0.0
    total_latency = 0.0

    for profile in all_profiles:
        # Determine capable models
        capable = list(model_costs.keys())
        if profile["needs_vision"]:
            capable = [m for m in capable if m in ("gpt-4o",)]  # only gpt-4o has vision
        if profile["needs_reasoning"]:
            capable = [m for m in capable if m not in ("qwen-3b-local",)]
        if profile["needs_tools"]:
            capable = [m for m in capable if m not in ("qwen-3b-local",)]

        if not capable:
            capable = ["gpt-4o-mini"]  # fallback

        # Select based on cost sensitivity
        cs = profile["cost_sensitivity"]
        if cs == "high":
            selected = min(capable, key=lambda m: model_costs[m]["output"])
        elif cs == "low":
            selected = max(capable, key=lambda m: model_costs[m]["quality"])
        else:
            # Balanced: minimize cost/quality ratio
            selected = min(capable, key=lambda m: model_costs[m]["output"] / model_costs[m]["quality"])

        model_selections.append(selected)
        # Estimate cost (500 input + 300 output tokens)
        mc = model_costs[selected]
        cost = (500/1000) * mc["input"] + (300/1000) * mc["output"]
        total_cost += cost
        total_latency += mc["latency"]

    # --- Report ---
    dist = Counter(model_selections)
    print(f"\n  Model Selection Distribution:")
    for m, count in dist.most_common():
        pct = count / len(model_selections) * 100
        print(f"    {m:<20} {count:>6} ({pct:>5.1f}%)")

    print(f"\n  Task Type Distribution:")
    tt_dist = Counter(p["task_type"] for p in all_profiles)
    for tt, count in tt_dist.most_common():
        pct = count / len(all_profiles) * 100
        print(f"    {tt:<20} {count:>6} ({pct:>5.1f}%)")

    print(f"\n  Cost Sensitivity Distribution:")
    cs_dist = Counter(p["cost_sensitivity"] for p in all_profiles)
    for cs, count in cs_dist.most_common():
        pct = count / len(all_profiles) * 100
        print(f"    {cs:<20} {count:>6} ({pct:>5.1f}%)")

    avg_cost = total_cost / len(model_selections)
    cost_per_1k = avg_cost * 1000
    avg_latency = total_latency / len(model_selections)

    # Compare to always-best (claude-sonnet) and always-cheapest (qwen-3b-local)
    always_best_cost = ((500/1000)*3.00 + (300/1000)*15.00) * 1000
    always_cheap_cost = ((500/1000)*0.001 + (300/1000)*0.003) * 1000

    print(f"\n  Cost Analysis:")
    print(f"    Router avg cost/query:   ${avg_cost:.6f}")
    print(f"    Router cost per 1K:      ${cost_per_1k:.4f}")
    print(f"    Always-best cost/1K:     ${always_best_cost:.4f} (claude-sonnet)")
    print(f"    Always-cheap cost/1K:    ${always_cheap_cost:.4f} (qwen-3b-local)")
    print(f"    Savings vs always-best:  {(1 - cost_per_1k/always_best_cost)*100:.1f}%")

    print(f"\n  Latency Analysis:")
    print(f"    Router avg latency:      {avg_latency:.0f}ms")

    # Reasoning/vision/tools stats
    reasoning_pct = sum(1 for p in all_profiles if p["needs_reasoning"]) / len(all_profiles) * 100
    vision_pct = sum(1 for p in all_profiles if p["needs_vision"]) / len(all_profiles) * 100
    tools_pct = sum(1 for p in all_profiles if p["needs_tools"]) / len(all_profiles) * 100
    avg_complexity = np.mean([p["complexity"] for p in all_profiles])
    avg_temp = np.mean([p["temperature"] for p in all_profiles])
    avg_thinking = np.mean([p["thinking_tokens"] for p in all_profiles])

    print(f"\n  Profile Statistics:")
    print(f"    Needs reasoning:         {reasoning_pct:.1f}%")
    print(f"    Needs vision:            {vision_pct:.1f}%")
    print(f"    Needs tools:             {tools_pct:.1f}%")
    print(f"    Avg complexity:          {avg_complexity:.4f}")
    print(f"    Avg temperature:         {avg_temp:.4f}")
    print(f"    Avg thinking tokens:     {avg_thinking:.0f}")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model: TaskProfileClassifier, metadata: dict):
    """Save classifier weights and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "classifier.pt")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Model saved to {MODEL_DIR}")


def load_saved_model(device: torch.device, input_dim: int = 384) -> TaskProfileClassifier:
    """Load a previously saved model."""
    model = TaskProfileClassifier(input_dim=input_dim)
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
    parser = argparse.ArgumentParser(description="Fine-tune task profile classifier on LMSYS 55K")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'mps', 'cpu', or 'auto'")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate a previously saved model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max training samples (0 = all)")
    parser.add_argument("--skip-routerarena", action="store_true",
                        help="Skip RouterArena evaluation")
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

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load LMSYS 55K data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading LMSYS 55K data...")
    prompts = load_lmsys_data(max_samples=args.max_samples)

    # ------------------------------------------------------------------
    # 2. Derive task profiles (rule-based labeling)
    # ------------------------------------------------------------------
    print("\n[2/6] Deriving task profiles for all queries...")
    t0 = time.time()
    profiles = [derive_task_profile(p) for p in prompts]
    print(f"  Derived {len(profiles)} profiles in {time.time()-t0:.1f}s")

    # Distribution report
    tt_dist = Counter(p["task_type"] for p in profiles)
    cs_dist = Counter(p["cost_sensitivity"] for p in profiles)
    reason_pct = sum(1 for p in profiles if p["needs_reasoning"]) / len(profiles) * 100

    print(f"\n  Task type distribution:")
    for tt, count in tt_dist.most_common():
        print(f"    {tt:<15} {count:>6} ({count/len(profiles)*100:.1f}%)")
    print(f"\n  Cost sensitivity distribution:")
    for cs, count in cs_dist.most_common():
        print(f"    {cs:<15} {count:>6} ({count/len(profiles)*100:.1f}%)")
    print(f"  Needs reasoning: {reason_pct:.1f}%")

    # ------------------------------------------------------------------
    # 3. Encode with MiniLM
    # ------------------------------------------------------------------
    print(f"\n[3/6] Encoding with {ENCODER_NAME}...")
    encoder = SentenceTransformer(ENCODER_NAME, device=str(device))
    emb_dim = (encoder.get_embedding_dimension()
               if hasattr(encoder, 'get_embedding_dimension')
               else encoder.get_sentence_embedding_dimension())
    print(f"  Embedding dim: {emb_dim}")
    embeddings = encode_texts(encoder, prompts)

    # ------------------------------------------------------------------
    # 4. Train/val split and prepare targets
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting data (85/15)...")
    indices = np.arange(len(prompts))
    train_idx, val_idx = train_test_split(indices, test_size=0.15,
                                           random_state=SEED)
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    profiles_train = [profiles[i] for i in train_idx]
    profiles_val = [profiles[i] for i in val_idx]

    targets_train = prepare_targets(profiles_train)
    targets_val = prepare_targets(profiles_val)

    print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    # ------------------------------------------------------------------
    # 5. Train or load model
    # ------------------------------------------------------------------
    if args.eval_only:
        print("\n[5/6] Loading saved model...")
        model = load_saved_model(device, emb_dim)
    else:
        print(f"\n[5/6] Training multi-head classifier (epochs={args.epochs})...")
        model = train_model(X_train, targets_train, X_val, targets_val,
                           device, epochs=args.epochs)

        metadata = {
            "encoder": ENCODER_NAME,
            "embedding_dim": emb_dim,
            "task_types": TASK_TYPES,
            "cost_labels": COST_LABELS,
            "hidden_dim": 256,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "total_samples": len(prompts),
            "epochs_config": args.epochs,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        }
        save_model(model, metadata)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluation...")
    eval_results = evaluate_model(model, X_val, targets_val, profiles_val,
                                   device, split_name="Validation (15%)")

    # RouterArena evaluation
    if not args.skip_routerarena:
        evaluate_on_routerarena(model, encoder, device)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time:                {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Training data:             {len(prompts)} LMSYS queries")
    print(f"  Encoder:                   {ENCODER_NAME}")
    print(f"  Classifier params:         {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Task type accuracy:        {eval_results['task_type_accuracy']:.4f}")
    print(f"  Cost sensitivity accuracy: {eval_results['cost_sensitivity_accuracy']:.4f}")
    print(f"  Needs reasoning accuracy:  {eval_results['needs_reasoning_accuracy']:.4f}")
    print(f"  Complexity MAE:            {eval_results['complexity_mae']:.4f}")
    print(f"  Temperature MAE:           {eval_results['temperature_mae']:.4f}")
    print(f"  Thinking tokens MAE:       {eval_results['thinking_tokens_mae']:.4f}")
    print(f"  Model saved to:            {MODEL_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
