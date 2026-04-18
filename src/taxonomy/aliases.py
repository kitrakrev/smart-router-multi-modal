"""Map variant names to canonical taxonomy paths."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical path -> list of known aliases (including the canonical name itself)
SPECIALTY_ALIASES: dict[str, list[str]] = {
    "pathology": [
        "pathology",
        "pathologies",
        "histopathology",
        "histology",
        "tissue analysis",
        "biopsy reading",
    ],
    "pathology.deep_lesions": [
        "deep lesion",
        "lesion analysis",
        "deep lesion detection",
    ],
    "radiology": [
        "radiology",
        "radiography",
        "imaging",
        "medical imaging",
        "x-ray interpretation",
    ],
    "radiology.chest": [
        "chest x-ray",
        "cxr",
        "thoracic imaging",
        "chest radiograph",
    ],
    "cardiology": [
        "cardiology",
        "cardiac",
        "heart",
        "cardiovascular",
    ],
    "dermatology": [
        "dermatology",
        "skin",
        "dermatological",
    ],
    "ophthalmology": [
        "ophthalmology",
        "eye",
        "retina",
        "fundus",
    ],
    "emergency": [
        "emergency",
        "trauma",
        "acute",
        "ER",
    ],
    "pharmacology": [
        "pharmacology",
        "drugs",
        "medication",
        "prescription",
    ],
    "general_medicine": [
        "general medicine",
        "primary care",
        "internal medicine",
    ],
}

# Build reverse lookup: alias -> canonical path
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SPECIALTY_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


class _FuzzyMatcher:
    """Embedding-based fuzzy matcher for unknown terms."""

    _instance: Optional["_FuzzyMatcher"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model = None
        self._alias_embeddings: dict[str, np.ndarray] = {}

    @classmethod
    async def get_instance(cls) -> "_FuzzyMatcher":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")

            # Embed all aliases, keyed by canonical path
            for canonical, aliases in SPECIALTY_ALIASES.items():
                embeddings = self._model.encode(aliases, convert_to_numpy=True)
                avg = np.mean(embeddings, axis=0)
                avg = avg / np.linalg.norm(avg)
                self._alias_embeddings[canonical] = avg

        except ImportError:
            logger.warning("sentence-transformers not installed. Fuzzy alias matching unavailable.")

    def match(self, term: str, threshold: float = 0.4) -> Optional[str]:
        """Find best matching canonical path for an unknown term."""
        if self._model is None:
            return None

        term_emb = self._model.encode(term, convert_to_numpy=True)
        term_emb = term_emb / np.linalg.norm(term_emb)

        best_canonical: Optional[str] = None
        best_score = threshold

        for canonical, emb in self._alias_embeddings.items():
            sim = float(np.dot(term_emb, emb))
            if sim > best_score:
                best_score = sim
                best_canonical = canonical

        return best_canonical


def resolve_alias(term: str) -> Optional[str]:
    """Resolve a specialty term to its canonical taxonomy path.

    First checks exact alias lookup, returns None if no match found.
    Use resolve_alias_fuzzy for embedding-based fallback.

    Args:
        term: A specialty name or alias.

    Returns:
        Canonical taxonomy path (e.g., "pathology") or None.
    """
    return _ALIAS_TO_CANONICAL.get(term.lower())


async def resolve_alias_fuzzy(term: str, threshold: float = 0.4) -> Optional[str]:
    """Resolve a specialty term using exact lookup + embedding fallback.

    Args:
        term: A specialty name or alias.
        threshold: Minimum cosine similarity for fuzzy match.

    Returns:
        Canonical taxonomy path or None.
    """
    # Try exact match first
    exact = resolve_alias(term)
    if exact is not None:
        return exact

    # Fallback to embedding-based fuzzy match
    matcher = await _FuzzyMatcher.get_instance()
    return matcher.match(term, threshold)
