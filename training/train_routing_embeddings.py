#!/usr/bin/env python3
"""
Train custom routing embeddings via contrastive fine-tuning of MiniLM-L6-v2.

Goal: queries needing the same routing decision cluster together in embedding
space.  This gives the router better cosine-similarity matching than generic
sentence embeddings.

Training data: LMSYS 55K (real user queries with derived task profiles).
NO RouterArena data is used (that is test-only).

Method: MultipleNegativesRankingLoss on (anchor, positive) pairs where both
queries share the same task_type and similar complexity.  Negatives are mined
from other pairs in the batch automatically.

Budget-aware: the embedding space naturally clusters by cost tier because
task types correlate with cost (simple_qa = cheap, complex_reasoning = expensive).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 train_routing_embeddings.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path(__file__).parent / "models" / "routing-embeddings"
SEED = 42
MAX_PAIRS = 60000       # cap training pairs
EVAL_FRACTION = 0.1     # 10% for eval
BATCH_SIZE = 512        # A100 80GB can handle this easily
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Task-type classification (same logic as finetune_lmsys.py)
# ---------------------------------------------------------------------------

TASK_TYPES = ["qa", "code", "math", "creative", "translation", "reasoning", "chat"]

_CODE_PATTERNS = [
    r'\b(write|implement|create|build|code|debug|fix|refactor)\b.*\b(function|class|program|script|code|app|api|server|module)\b',
    r'\b(python|javascript|java|c\+\+|rust|go|typescript|html|css|sql|bash|ruby|swift|kotlin)\b',
    r'\b(import|def |class |function |const |let |var |public |private )\b',
    r'\b(bug|error|exception|traceback|stack\s*trace|compile|runtime)\b',
    r'\b(algorithm|data\s*structure|binary\s*search|sort|hash|tree|graph|linked\s*list)\b',
    r'```',
]

_MATH_PATTERNS = [
    r'\b(solve|prove|derive|calculate|compute|evaluate|simplify|factor)\b',
    r'\b(equation|integral|derivative|matrix|vector|eigenvalue|polynomial)\b',
    r'\b(theorem|lemma|corollary|proof|qed|contradiction)\b',
    r'\b(algebra|calculus|geometry|trigonometry|statistics|probability)\b',
    r'\b(sin|cos|tan|log|ln|exp|sqrt|lim|sum|product)\b[\s(]',
    r'[=+\-*/^]\s*\d+.*[=+\-*/^]',
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

_TECHNICAL_TERMS = {
    "algorithm", "optimization", "architecture", "distributed", "concurrent",
    "asymptotic", "polynomial", "differential", "stochastic", "recurrence",
    "eigenvalue", "topology", "morphism", "isomorphism", "gradient",
    "backpropagation", "integral", "derivative", "theorem", "convergence",
    "fourier", "laplace", "cryptography", "recursion", "heuristic",
    "quantum", "neural", "transformer", "attention", "embedding",
    "manifold", "convex", "entropy", "markov", "tensor", "jacobian",
}


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    count = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return count


def estimate_complexity(text: str) -> float:
    words = text.lower().split()
    word_count = len(words)
    length_score = min(word_count / 200, 1.0)
    tech_count = sum(1 for w in words if w.strip("?.,!:;()[]{}") in _TECHNICAL_TERMS)
    tech_score = min(tech_count / 4, 1.0)
    sentence_count = max(text.count(".") + text.count("?") + text.count("!"), 1)
    multi_step = min(sentence_count / 5, 1.0)
    has_code_block = 1.0 if "```" in text else 0.0
    q_words = {"what", "why", "how", "explain", "describe", "compare",
               "analyze", "evaluate", "derive", "prove", "justify"}
    q_count = sum(1 for w in words if w.strip("?.,!") in q_words)
    q_score = min(q_count / 3, 1.0)
    score = (0.15 * length_score + 0.30 * tech_score + 0.15 * multi_step +
             0.10 * has_code_block + 0.05 * 0.0 + 0.25 * q_score)
    return round(min(max(score, 0.0), 1.0), 4)


def classify_task_type(text: str) -> str:
    text_lower = text.lower()
    words = text_lower.split()
    scores = {
        "code": _count_pattern_matches(text, _CODE_PATTERNS),
        "math": _count_pattern_matches(text, _MATH_PATTERNS),
        "creative": _count_pattern_matches(text, _CREATIVE_PATTERNS),
        "translation": _count_pattern_matches(text, _TRANSLATION_PATTERNS),
        "reasoning": _count_pattern_matches(text, _REASONING_PATTERNS),
        "qa": _count_pattern_matches(text, _QA_PATTERNS),
    }
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "chat"
    elif scores[best_type] <= 1 and len(words) < 10:
        return "chat"
    return best_type


# Map task types to routing decisions (same as config.yaml decisions)
TASK_TO_DECISION = {
    "qa": "simple_qa",
    "code": "code_generation",
    "math": "complex_reasoning",
    "creative": "creative_writing",
    "translation": "general",       # translation -> general (cheapest)
    "reasoning": "complex_reasoning",
    "chat": "general",
}

# Cost tier for each decision (used to create budget-aware clusters)
DECISION_COST_TIER = {
    "simple_qa": 0,       # cheapest
    "general": 0,         # cheapest
    "creative_writing": 1, # cheap-mid
    "code_generation": 1,  # mid
    "complex_reasoning": 2, # most expensive
    "vision_task": 3,      # special (requires vision model)
    "tool_usage": 3,       # special (requires tool-capable)
}


# ---------------------------------------------------------------------------
# Load and process LMSYS 55K data
# ---------------------------------------------------------------------------

def load_lmsys_queries() -> list[dict]:
    """Load LMSYS Arena 55K and derive task profiles for each query."""
    print("Loading LMSYS Arena 55K from HuggingFace...")
    ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
    print(f"  Loaded {len(ds)} conversations")

    queries = []
    for i, row in enumerate(ds):
        # Extract first user turn
        prompt = ""
        if "prompt" in row and row["prompt"]:
            prompt = row["prompt"]
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
                    if isinstance(parsed, list) and len(parsed) > 0:
                        prompt = parsed[0].get("content", "") if isinstance(parsed[0], dict) else str(parsed[0])
                except (json.JSONDecodeError, TypeError):
                    prompt = conv[:500]

        if not prompt or len(prompt.strip()) < 5:
            continue

        # Truncate very long prompts
        if len(prompt) > 1000:
            prompt = prompt[:1000]

        task_type = classify_task_type(prompt)
        complexity = estimate_complexity(prompt)
        decision = TASK_TO_DECISION.get(task_type, "general")
        cost_tier = DECISION_COST_TIER.get(decision, 0)

        queries.append({
            "text": prompt.strip(),
            "task_type": task_type,
            "decision": decision,
            "complexity": complexity,
            "cost_tier": cost_tier,
        })

    print(f"  Extracted {len(queries)} valid queries")
    return queries


# ---------------------------------------------------------------------------
# Generate contrastive pairs
# ---------------------------------------------------------------------------

def generate_pairs(queries: list[dict]) -> tuple[list[InputExample], list[InputExample]]:
    """Generate (anchor, positive) pairs for MultipleNegativesRankingLoss.

    Positive pairs: same decision AND similar complexity (within 0.3).
    The loss function automatically treats other in-batch pairs as negatives.

    Also generates exemplar-based pairs from config decision definitions.
    """
    # Group queries by decision
    by_decision: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_decision[q["decision"]].append(q)

    print("\nQuery distribution by decision:")
    for dec, qs in sorted(by_decision.items()):
        print(f"  {dec}: {len(qs)}")

    pairs = []

    # ---- Pairs from LMSYS queries ----
    for decision, group in by_decision.items():
        if len(group) < 2:
            continue

        # Sort by complexity for efficient pair generation
        group_sorted = sorted(group, key=lambda x: x["complexity"])

        # Sliding window: pair queries with similar complexity
        for i in range(len(group_sorted)):
            # Look at nearby queries (complexity-wise)
            for j in range(i + 1, min(i + 10, len(group_sorted))):
                if abs(group_sorted[i]["complexity"] - group_sorted[j]["complexity"]) <= 0.3:
                    pairs.append(InputExample(
                        texts=[group_sorted[i]["text"], group_sorted[j]["text"]]
                    ))

    print(f"\nGenerated {len(pairs)} pairs from LMSYS queries")

    # ---- Pairs from decision exemplars (config.yaml style) ----
    exemplar_pairs = _generate_exemplar_pairs()
    pairs.extend(exemplar_pairs)
    print(f"Generated {len(exemplar_pairs)} pairs from decision exemplars")

    # Shuffle and cap
    random.shuffle(pairs)
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    print(f"Total training pairs (capped): {len(pairs)}")

    # Split into train/eval
    split_idx = int(len(pairs) * (1 - EVAL_FRACTION))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")
    return train_pairs, eval_pairs


def _generate_exemplar_pairs() -> list[InputExample]:
    """Generate pairs from decision exemplars (like config.yaml decisions)."""
    # These exemplars define what each decision cluster should look like.
    # Every pair within the same decision is a positive pair.
    decision_exemplars = {
        "complex_reasoning": [
            "prove that the square root of 2 is irrational",
            "derive the quadratic formula from first principles",
            "solve this system of differential equations",
            "find the eigenvalues and eigenvectors of matrix A",
            "prove by induction that the sum of first n integers equals n(n+1)/2",
            "analyze the time complexity of this algorithm",
            "what is the limit of this sequence as n approaches infinity",
            "find the general solution to this recurrence relation",
            "explain why the halting problem is undecidable",
            "prove the fundamental theorem of calculus",
            "solve this optimization problem using Lagrange multipliers",
            "what is the computational complexity of this NP-hard problem",
        ],
        "code_generation": [
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
            "write a React component for a login form",
            "create a Dockerfile for this Node.js app",
        ],
        "creative_writing": [
            "write a short story about a time traveler",
            "compose a haiku about autumn",
            "write a compelling product description",
            "create a dialogue between two historical figures",
            "write a poem about the ocean",
            "brainstorm 10 startup ideas for AI",
            "write a persuasive essay about climate change",
            "come up with a creative name for my app",
            "write a fairy tale for children",
            "compose a love letter in Victorian style",
        ],
        "simple_qa": [
            "what is the capital of France",
            "who invented the telephone",
            "how many planets are in the solar system",
            "what year did World War 2 end",
            "define photosynthesis",
            "what is the boiling point of water",
            "who wrote Romeo and Juliet",
            "what is the largest ocean on earth",
            "when was the United Nations founded",
            "what is the speed of light",
        ],
        "general": [
            "tell me about yourself",
            "what can you help me with",
            "summarize this article",
            "rewrite this email to be more professional",
            "explain this concept in simple terms",
            "give me advice on time management",
            "compare these two approaches",
            "help me plan my vacation",
            "what are the best practices for remote work",
            "how do I improve my resume",
        ],
        "vision_task": [
            "what is shown in this image",
            "describe the chart and extract the data",
            "read the text in this screenshot",
            "analyze this diagram",
            "what breed is this dog in the photo",
            "compare these two images",
            "identify the objects in this picture",
            "what does this graph show",
        ],
        "tool_usage": [
            "search the web for latest news about AI",
            "calculate 15% tip on $47.50",
            "look up the weather in San Francisco",
            "run a vulnerability scan on this server",
            "execute a port scan on the target",
            "check the stock price of NVIDIA",
            "find restaurants near me",
            "schedule a meeting for tomorrow at 3pm",
        ],
    }

    pairs = []
    for decision, exemplars in decision_exemplars.items():
        # All pairs within each decision are positive
        for i in range(len(exemplars)):
            for j in range(i + 1, len(exemplars)):
                pairs.append(InputExample(
                    texts=[exemplars[i], exemplars[j]]
                ))

    return pairs


# ---------------------------------------------------------------------------
# Build eval data for EmbeddingSimilarityEvaluator
# ---------------------------------------------------------------------------

def build_similarity_eval(eval_pairs: list[InputExample]) -> EmbeddingSimilarityEvaluator:
    """Build evaluator from eval pairs.

    For MNRL, we create similarity labels: same-decision pairs get score 1.0.
    We also add cross-decision negative pairs with score 0.0.
    """
    sentences1 = []
    sentences2 = []
    scores = []

    for pair in eval_pairs:
        sentences1.append(pair.texts[0])
        sentences2.append(pair.texts[1])
        scores.append(1.0)  # positive pair

    # Add some negative pairs (random cross-pair)
    n_neg = min(len(eval_pairs), 500)
    for _ in range(n_neg):
        i = random.randint(0, len(eval_pairs) - 1)
        j = random.randint(0, len(eval_pairs) - 1)
        if i != j:
            sentences1.append(eval_pairs[i].texts[0])
            sentences2.append(eval_pairs[j].texts[1])
            scores.append(0.0)

    return EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, name="routing-eval")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0)
        vram = getattr(mem, 'total_memory', getattr(mem, 'total_mem', 0))
        print(f"VRAM: {vram / 1e9:.1f} GB")

    # 1. Load data
    queries = load_lmsys_queries()

    # 2. Generate contrastive pairs
    train_pairs, eval_pairs = generate_pairs(queries)

    # 3. Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device=device)
    print(f"  Model dimensions: {model.get_sentence_embedding_dimension()}")

    # 4. Set up training
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    evaluator = build_similarity_eval(eval_pairs)

    warmup_steps = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)
    print(f"\nTraining config:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Training batches/epoch: {len(train_dataloader)}")
    print(f"  Loss: MultipleNegativesRankingLoss")
    print(f"  Output: {OUTPUT_DIR}")

    # 5. Train
    print(f"\n{'='*60}")
    print(f"  TRAINING START")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=100,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_DIR),
        optimizer_params={"lr": LR},
        use_amp=True,  # bf16/fp16 on A100
        show_progress_bar=True,
        save_best_model=True,
    )

    t_train = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Duration: {t_train:.1f}s ({t_train/60:.1f} min)")
    print(f"  Model saved to: {OUTPUT_DIR}")

    # 6. Validate: compute cluster separability
    print(f"\n{'='*60}")
    print(f"  VALIDATION: Cluster Quality")
    print(f"{'='*60}")

    trained_model = SentenceTransformer(str(OUTPUT_DIR), device=device)
    base_model_val = SentenceTransformer(BASE_MODEL, device=device)

    # Compute centroids for each decision using exemplar texts
    decision_exemplars = {
        "complex_reasoning": [
            "prove that the square root of 2 is irrational",
            "analyze the time complexity of this algorithm",
            "solve this optimization problem using Lagrange multipliers",
        ],
        "code_generation": [
            "write a Python function to sort a list",
            "implement a binary search tree in Java",
            "create a REST API endpoint in FastAPI",
        ],
        "simple_qa": [
            "what is the capital of France",
            "how many planets are in the solar system",
            "who wrote Romeo and Juliet",
        ],
        "general": [
            "summarize this article",
            "rewrite this email to be more professional",
            "explain this concept in simple terms",
        ],
        "creative_writing": [
            "write a short story about a time traveler",
            "compose a haiku about autumn",
            "write a poem about the ocean",
        ],
    }

    test_queries = {
        "complex_reasoning": [
            "derive the formula for the volume of a sphere using integration",
            "prove that there are infinitely many prime numbers",
        ],
        "code_generation": [
            "write a Python class for a linked list with insert and delete methods",
            "debug this JavaScript async/await code that hangs",
        ],
        "simple_qa": [
            "what is the tallest mountain in the world",
            "who painted the Mona Lisa",
        ],
        "general": [
            "help me write a better cover letter",
            "give me tips for public speaking",
        ],
        "creative_writing": [
            "write a limerick about a programmer",
            "create a short sci-fi story about Mars colonization",
        ],
    }

    for model_name, eval_model in [("Base MiniLM", base_model_val), ("Trained", trained_model)]:
        print(f"\n  [{model_name}]")

        # Compute centroids
        centroids = {}
        for dec, exemplars in decision_exemplars.items():
            embs = eval_model.encode(exemplars)
            centroid = np.mean(embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[dec] = centroid

        # Test: for each test query, find closest centroid
        correct = 0
        total = 0
        for expected_dec, test_qs in test_queries.items():
            for q in test_qs:
                emb = eval_model.encode(q)
                emb = emb / np.linalg.norm(emb)
                sims = {d: float(np.dot(emb, c)) for d, c in centroids.items()}
                best = max(sims, key=sims.get)
                is_correct = best == expected_dec
                correct += int(is_correct)
                total += 1
                mark = "OK" if is_correct else "WRONG"
                print(f"    [{mark}] '{q[:50]}...' -> {best} (sim={sims[best]:.3f})"
                      f" expected={expected_dec}")

        print(f"  Classification accuracy: {correct}/{total} ({correct/total:.0%})")

        # Inter-cluster distances
        print(f"  Inter-cluster cosine similarities:")
        decisions_list = sorted(centroids.keys())
        for i, d1 in enumerate(decisions_list):
            for j, d2 in enumerate(decisions_list):
                if j > i:
                    sim = float(np.dot(centroids[d1], centroids[d2]))
                    print(f"    {d1} <-> {d2}: {sim:.3f}")

    # 7. Report
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Eval pairs: {len(eval_pairs)}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training time: {t_train:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Ready for: router.py DecisionMatcher, HuggingFace push")


if __name__ == "__main__":
    main()
