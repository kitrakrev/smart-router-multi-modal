"""Complexity signal using contrastive hard/easy exemplars (vLLM-SR style)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Hard exemplars: queries that require deep reasoning, multi-step analysis
HARD_EXEMPLARS: list[str] = [
    "prove the theorem that every continuous function on a closed interval is uniformly continuous",
    "differential diagnosis for a 45-year-old male with acute chest pain radiating to the left arm with diaphoresis",
    "analyze the mechanism of action of immune checkpoint inhibitors in the tumor microenvironment",
    "compare and contrast the pathophysiology of type 1 and type 2 diabetes at the molecular level",
    "explain the hemodynamic changes in cardiogenic shock and outline a step-by-step management algorithm",
    "derive the pharmacokinetic equations for a two-compartment model with first-order elimination",
    "critically appraise this randomized controlled trial design and identify potential sources of bias",
    "explain the molecular basis of antibiotic resistance mechanisms in gram-negative bacteria",
    "describe the complete coagulation cascade and how each anticoagulant class intervenes",
    "analyze the risk-benefit of immunosuppressive therapy in transplant patients with active infection",
]

# Easy exemplars: simple factual queries, definitions, yes/no answers
EASY_EXEMPLARS: list[str] = [
    "what is normal blood pressure",
    "define the term hypertension",
    "yes or no: is aspirin a blood thinner",
    "what does MRI stand for",
    "name three symptoms of the common cold",
    "what is the normal body temperature",
    "is ibuprofen an NSAID",
    "what organ produces insulin",
    "how many chambers does the heart have",
    "what is the abbreviation for complete blood count",
]


@dataclass
class ComplexitySignalResult:
    """Result from complexity signal analysis."""

    complexity_score: float  # 0.0 (easy) to 1.0 (hard)
    hard_similarity: float
    easy_similarity: float
    label: str  # "easy", "medium", "hard"
    error: Optional[str] = None


class ComplexitySignalModel:
    """Lazy-loaded embedding model for contrastive complexity estimation."""

    _instance: Optional["ComplexitySignalModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model = None
        self._hard_embedding: Optional[np.ndarray] = None
        self._easy_embedding: Optional[np.ndarray] = None

    @classmethod
    async def get_instance(cls) -> "ComplexitySignalModel":
        """Get or create singleton instance with lazy loading."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self) -> None:
        """Load the embedding model and pre-compute exemplar centroids."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous model loading and exemplar embedding."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded all-MiniLM-L6-v2 for complexity signal")

            # Compute centroid of hard exemplars
            hard_embeddings = self._model.encode(HARD_EXEMPLARS, convert_to_numpy=True)
            self._hard_embedding = np.mean(hard_embeddings, axis=0)
            self._hard_embedding = self._hard_embedding / np.linalg.norm(self._hard_embedding)

            # Compute centroid of easy exemplars
            easy_embeddings = self._model.encode(EASY_EXEMPLARS, convert_to_numpy=True)
            self._easy_embedding = np.mean(easy_embeddings, axis=0)
            self._easy_embedding = self._easy_embedding / np.linalg.norm(self._easy_embedding)

        except ImportError:
            logger.error("sentence-transformers not installed. Complexity signal unavailable.")

    async def estimate_complexity(self, messages: list[dict]) -> ComplexitySignalResult:
        """Estimate query complexity using contrastive cosine similarity."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._estimate_sync, messages)

    def _estimate_sync(self, messages: list[dict]) -> ComplexitySignalResult:
        """Synchronous complexity estimation."""
        if self._model is None or self._hard_embedding is None or self._easy_embedding is None:
            return ComplexitySignalResult(
                complexity_score=0.5,
                hard_similarity=0.0,
                easy_similarity=0.0,
                label="medium",
                error="Complexity model not loaded",
            )

        try:
            query_text = _extract_query_text(messages)
            if not query_text:
                return ComplexitySignalResult(
                    complexity_score=0.5,
                    hard_similarity=0.0,
                    easy_similarity=0.0,
                    label="medium",
                    error="No text content found",
                )

            # Embed query
            query_embedding = self._model.encode(query_text, convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Cosine similarity to hard and easy centroids
            hard_sim = float(np.dot(query_embedding, self._hard_embedding))
            easy_sim = float(np.dot(query_embedding, self._easy_embedding))

            # Normalized complexity score: (hard_sim - easy_sim + 1) / 2
            # Maps [-1, 1] range to [0, 1]
            complexity_score = (hard_sim - easy_sim + 1.0) / 2.0
            complexity_score = max(0.0, min(1.0, complexity_score))

            # Label assignment
            if complexity_score >= 0.65:
                label = "hard"
            elif complexity_score >= 0.40:
                label = "medium"
            else:
                label = "easy"

            return ComplexitySignalResult(
                complexity_score=complexity_score,
                hard_similarity=hard_sim,
                easy_similarity=easy_sim,
                label=label,
            )

        except Exception as e:
            logger.error(f"Complexity signal error: {e}")
            return ComplexitySignalResult(
                complexity_score=0.5,
                hard_similarity=0.0,
                easy_similarity=0.0,
                label="medium",
                error=str(e),
            )


def _extract_query_text(messages: list[dict]) -> str:
    """Extract text content from the last user message."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return " ".join(text_parts)
    return ""


async def complexity_signal(messages: list[dict]) -> ComplexitySignalResult:
    """Run the complexity signal: contrastive hard vs easy embedding similarity.

    Args:
        messages: OpenAI-format message list.

    Returns:
        ComplexitySignalResult with complexity_score (0-1), similarities, and label.
    """
    model = await ComplexitySignalModel.get_instance()
    return await model.estimate_complexity(messages)
