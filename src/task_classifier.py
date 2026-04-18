#!/usr/bin/env python3
"""
Task Profile Classifier - Inference wrapper.

Loads the trained multi-head model from finetune_lmsys.py and returns
task profiles for queries. Designed to be imported by signals.py as
a new signal source.

Usage as a signal:
    from task_classifier import task_profile_signal
    result = await task_profile_signal(messages)

Usage standalone:
    from task_classifier import TaskClassifier
    classifier = TaskClassifier()
    profile = classifier.predict("Prove that sqrt(2) is irrational")
    # => TaskProfile(task_type='math', complexity=0.72, needs_reasoning=True, ...)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (must match finetune_lmsys.py)
# ---------------------------------------------------------------------------
TASK_TYPES = ["qa", "code", "math", "creative", "translation", "reasoning", "chat"]
COST_LABELS = ["low", "medium", "high"]

MODEL_DIR = Path(__file__).parent.parent / "models" / "lmsys-task-classifier"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaskProfile:
    """Complete task profile predicted by the classifier."""
    task_type: str                   # one of TASK_TYPES
    complexity: float                # 0.0 - 1.0
    needs_reasoning: bool
    needs_vision: bool
    needs_tools: bool
    temperature: float               # 0.0 - 1.0
    max_tokens: int                  # 128 - 4096
    thinking_tokens: int             # 0 - 2000
    cost_sensitivity: str            # "low", "medium", "high"
    task_type_probs: dict = field(default_factory=dict)
    cost_sensitivity_probs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "complexity": round(self.complexity, 4),
            "needs_reasoning": self.needs_reasoning,
            "needs_vision": self.needs_vision,
            "needs_tools": self.needs_tools,
            "temperature": round(self.temperature, 3),
            "max_tokens": self.max_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_sensitivity": self.cost_sensitivity,
            "task_type_probs": {k: round(v, 4) for k, v in self.task_type_probs.items()},
            "cost_sensitivity_probs": {k: round(v, 4) for k, v in self.cost_sensitivity_probs.items()},
        }

    def get_inference_config(self) -> dict:
        """Build an inference configuration dict for the LLM call."""
        config = {
            "temperature": round(self.temperature, 2),
            "max_tokens": self.max_tokens,
        }
        if self.needs_reasoning and self.thinking_tokens > 0:
            config["enable_reasoning"] = True
            config["thinking_tokens"] = self.thinking_tokens
        return config

    def get_required_capabilities(self) -> set[str]:
        """Return required model capabilities for capability-based routing."""
        caps = {"text"}
        if self.needs_vision:
            caps.add("vision")
        if self.needs_tools:
            caps.add("tools")
        if self.needs_reasoning:
            caps.add("reasoning")
        if self.task_type == "code":
            caps.add("code")
        return caps


# ---------------------------------------------------------------------------
# Model architecture (must match finetune_lmsys.py)
# ---------------------------------------------------------------------------

class TaskProfileClassifierNet(torch.nn.Module):
    """Multi-head classifier matching the architecture in finetune_lmsys.py."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.task_type_head = torch.nn.Linear(hidden_dim, len(TASK_TYPES))
        self.cost_sensitivity_head = torch.nn.Linear(hidden_dim, len(COST_LABELS))
        self.complexity_head = torch.nn.Linear(hidden_dim, 1)
        self.needs_reasoning_head = torch.nn.Linear(hidden_dim, 1)
        self.needs_vision_head = torch.nn.Linear(hidden_dim, 1)
        self.needs_tools_head = torch.nn.Linear(hidden_dim, 1)
        self.temperature_head = torch.nn.Linear(hidden_dim, 1)
        self.max_tokens_head = torch.nn.Linear(hidden_dim, 1)
        self.thinking_tokens_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "task_type": self.task_type_head(h),
            "cost_sensitivity": self.cost_sensitivity_head(h),
            "complexity": torch.sigmoid(self.complexity_head(h)).squeeze(-1),
            "needs_reasoning": torch.sigmoid(self.needs_reasoning_head(h)).squeeze(-1),
            "needs_vision": torch.sigmoid(self.needs_vision_head(h)).squeeze(-1),
            "needs_tools": torch.sigmoid(self.needs_tools_head(h)).squeeze(-1),
            "temperature": torch.sigmoid(self.temperature_head(h)).squeeze(-1),
            "max_tokens": torch.sigmoid(self.max_tokens_head(h)).squeeze(-1),
            "thinking_tokens": torch.nn.functional.relu(self.thinking_tokens_head(h)).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Classifier wrapper (singleton pattern)
# ---------------------------------------------------------------------------

_classifier_instance: Optional["TaskClassifier"] = None


class TaskClassifier:
    """
    Inference wrapper for the trained task profile classifier.

    Loads the model lazily on first use and caches it as a singleton.
    Thread-safe for read-only inference.
    """

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "auto"):
        self.model_dir = model_dir
        self._device = self._resolve_device(device)
        self._model: Optional[TaskProfileClassifierNet] = None
        self._encoder = None
        self._metadata: dict = {}
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _ensure_loaded(self):
        """Lazy load model and encoder on first use."""
        if self._loaded:
            return

        # Load metadata
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        input_dim = self._metadata.get("embedding_dim", 384)
        hidden_dim = self._metadata.get("hidden_dim", 256)

        # Load model
        self._model = TaskProfileClassifierNet(input_dim=input_dim, hidden_dim=hidden_dim)
        state = torch.load(
            self.model_dir / "classifier.pt",
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

        # Load encoder
        encoder_name = self._metadata.get("encoder", ENCODER_NAME)
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(encoder_name, device=str(self._device))

        self._loaded = True

    def predict(self, text: str) -> TaskProfile:
        """Predict task profile for a single query string."""
        self._ensure_loaded()
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[TaskProfile]:
        """Predict task profiles for a batch of queries."""
        self._ensure_loaded()

        # Encode
        embeddings = self._encoder.encode(
            texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Inference
        x = torch.tensor(embeddings, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            outputs = self._model(x)

        # Convert outputs to TaskProfile objects
        profiles = []
        task_type_probs = torch.softmax(outputs["task_type"], dim=1).cpu().numpy()
        cost_probs = torch.softmax(outputs["cost_sensitivity"], dim=1).cpu().numpy()

        for i in range(len(texts)):
            # Task type
            tt_idx = outputs["task_type"][i].argmax().item()
            tt_prob_dict = {TASK_TYPES[j]: float(task_type_probs[i][j]) for j in range(len(TASK_TYPES))}

            # Cost sensitivity
            cs_idx = outputs["cost_sensitivity"][i].argmax().item()
            cs_prob_dict = {COST_LABELS[j]: float(cost_probs[i][j]) for j in range(len(COST_LABELS))}

            # Rescale continuous outputs
            max_tokens_raw = outputs["max_tokens"][i].item()
            max_tokens = max(128, min(4096, int(max_tokens_raw * 4096)))
            # Round to nearest power-of-2-ish value
            token_steps = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]
            max_tokens = min(token_steps, key=lambda t: abs(t - max_tokens))

            thinking_raw = outputs["thinking_tokens"][i].item()
            thinking_tokens = max(0, min(2000, int(thinking_raw * 2000)))
            # Round to nearest 100
            thinking_tokens = round(thinking_tokens / 100) * 100

            profile = TaskProfile(
                task_type=TASK_TYPES[tt_idx],
                complexity=round(outputs["complexity"][i].item(), 4),
                needs_reasoning=outputs["needs_reasoning"][i].item() > 0.5,
                needs_vision=outputs["needs_vision"][i].item() > 0.5,
                needs_tools=outputs["needs_tools"][i].item() > 0.5,
                temperature=round(outputs["temperature"][i].item(), 3),
                max_tokens=max_tokens,
                thinking_tokens=thinking_tokens,
                cost_sensitivity=COST_LABELS[cs_idx],
                task_type_probs=tt_prob_dict,
                cost_sensitivity_probs=cs_prob_dict,
            )
            profiles.append(profile)

        return profiles

    def predict_from_embedding(self, embedding: np.ndarray) -> TaskProfile:
        """Predict from a pre-computed embedding (384-dim)."""
        self._ensure_loaded()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        x = torch.tensor(embedding, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            outputs = self._model(x)

        tt_idx = outputs["task_type"][0].argmax().item()
        cs_idx = outputs["cost_sensitivity"][0].argmax().item()

        task_type_probs = torch.softmax(outputs["task_type"], dim=1).cpu().numpy()[0]
        cost_probs = torch.softmax(outputs["cost_sensitivity"], dim=1).cpu().numpy()[0]

        max_tokens_raw = outputs["max_tokens"][0].item()
        max_tokens = max(128, min(4096, int(max_tokens_raw * 4096)))
        token_steps = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]
        max_tokens = min(token_steps, key=lambda t: abs(t - max_tokens))

        thinking_raw = outputs["thinking_tokens"][0].item()
        thinking_tokens = max(0, min(2000, int(thinking_raw * 2000)))
        thinking_tokens = round(thinking_tokens / 100) * 100

        return TaskProfile(
            task_type=TASK_TYPES[tt_idx],
            complexity=round(outputs["complexity"][0].item(), 4),
            needs_reasoning=outputs["needs_reasoning"][0].item() > 0.5,
            needs_vision=outputs["needs_vision"][0].item() > 0.5,
            needs_tools=outputs["needs_tools"][0].item() > 0.5,
            temperature=round(outputs["temperature"][0].item(), 3),
            max_tokens=max_tokens,
            thinking_tokens=thinking_tokens,
            cost_sensitivity=COST_LABELS[cs_idx],
            task_type_probs={TASK_TYPES[j]: float(task_type_probs[j]) for j in range(len(TASK_TYPES))},
            cost_sensitivity_probs={COST_LABELS[j]: float(cost_probs[j]) for j in range(len(COST_LABELS))},
        )


def get_classifier(device: str = "auto") -> TaskClassifier:
    """Get or create the singleton TaskClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TaskClassifier(device=device)
    return _classifier_instance


# ---------------------------------------------------------------------------
# Signal interface (for integration with signals.py)
# ---------------------------------------------------------------------------

async def task_profile_signal(messages: list[dict], **_kw) -> "SignalResult":
    """
    Signal that returns the ML-predicted task profile.

    Returns a SignalResult with the full task profile in metadata.
    Score = complexity, confidence = max task_type probability.
    """
    from signals import SignalResult

    t0 = time.perf_counter()

    # Extract text from messages
    parts = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
    text = " ".join(parts)

    if not text.strip():
        return SignalResult(
            name="task_profile",
            score=0.0,
            confidence=0.0,
            execution_time_ms=0,
            metadata={"error": "empty query"},
            skipped=True,
        )

    try:
        classifier = get_classifier()
        profile = classifier.predict(text)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SignalResult(
            name="task_profile",
            score=profile.complexity,
            confidence=max(profile.task_type_probs.values()) if profile.task_type_probs else 0.5,
            execution_time_ms=elapsed_ms,
            metadata=profile.to_dict(),
        )
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return SignalResult(
            name="task_profile",
            score=0.0,
            confidence=0.0,
            execution_time_ms=elapsed_ms,
            metadata={"error": str(e)},
            skipped=True,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    queries = [
        "What is the capital of France?",
        "Write a Python function to implement quicksort",
        "Prove that the square root of 2 is irrational",
        "Write a poem about the ocean at sunset",
        "Translate this sentence to French: The weather is beautiful today",
        "Compare the pros and cons of React vs Vue for a large enterprise app",
        "Hi, how are you doing today?",
        "Search the web for the latest news about AI regulation",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    classifier = TaskClassifier()
    print(f"Device: {classifier._device}")
    print()

    for q in queries:
        t0 = time.perf_counter()
        profile = classifier.predict(q)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Query: {q[:80]}...")
        print(f"  Task type:        {profile.task_type} (p={profile.task_type_probs.get(profile.task_type, 0):.3f})")
        print(f"  Complexity:       {profile.complexity:.4f}")
        print(f"  Needs reasoning:  {profile.needs_reasoning}")
        print(f"  Needs vision:     {profile.needs_vision}")
        print(f"  Needs tools:      {profile.needs_tools}")
        print(f"  Temperature:      {profile.temperature:.3f}")
        print(f"  Max tokens:       {profile.max_tokens}")
        print(f"  Thinking tokens:  {profile.thinking_tokens}")
        print(f"  Cost sensitivity: {profile.cost_sensitivity}")
        print(f"  Required caps:    {profile.get_required_capabilities()}")
        print(f"  Inference config: {profile.get_inference_config()}")
        print(f"  Latency:          {elapsed:.1f}ms")
        print()
