"""Text embedding signal for specialty matching."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# Specialty exemplar queries for embedding-based matching
SPECIALTY_EXEMPLARS: dict[str, list[str]] = {
    "medical.radiology": [
        "interpret this chest X-ray for pneumonia",
        "CT scan findings suggest pulmonary embolism",
        "MRI brain showing white matter lesions",
        "radiographic assessment of fracture",
    ],
    "medical.pathology": [
        "histological features of adenocarcinoma",
        "tissue biopsy showing cellular atypia and invasion",
        "immunohistochemistry staining pattern interpretation",
        "grade this tumor using WHO classification",
    ],
    "medical.dermatology": [
        "pigmented skin lesion with irregular borders",
        "rash differential diagnosis with distribution pattern",
        "dermoscopic features of melanoma vs nevus",
        "psoriasis treatment options and severity grading",
    ],
    "medical.general_medicine": [
        "routine health screening recommendations",
        "hypertension initial workup and management",
        "diabetes type 2 lifestyle modifications",
        "preventive care guidelines for adults",
    ],
    "general.code": [
        "write a Python function to sort a list",
        "debug this JavaScript async await code",
        "implement a REST API endpoint",
        "refactor this class for better performance",
    ],
    "general.reasoning": [
        "explain why this logical argument is valid",
        "analyze the pros and cons of this approach",
        "solve this mathematical optimization problem",
        "prove this theorem using induction",
    ],
    "general.creative": [
        "write a short story about time travel",
        "compose a poem about nature",
        "brainstorm marketing taglines for a product",
        "create a fictional dialogue between characters",
    ],
    "general.simple_qa": [
        "what is the capital of France",
        "how many planets are in the solar system",
        "define photosynthesis",
        "who invented the telephone",
    ],
}


@dataclass
class TextSignalResult:
    """Result from text embedding specialty matching."""

    matched_specialty: str
    similarity: float
    all_scores: dict[str, float] = field(default_factory=dict)
    is_medical: bool = False
    error: Optional[str] = None


class TextSignalModel:
    """Dual-embedding model: medical (BioLORD) + general (BGE-small) ensemble.

    Runs both models, takes max similarity per specialty. Medical exemplars
    score higher on BioLORD, general exemplars on BGE. Falls back to
    MiniLM-L6-v2 if neither BioLORD nor BGE available.
    """

    _instance: Optional["TextSignalModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    # Model priority: try best first, fall back
    # Try trained model first, fall back to off-the-shelf
    TRAINED_MODEL = "trained_router_v2"  # local path, fine-tuned BioLORD on 9K medical dataset
    MEDICAL_MODEL = "FremyCompany/BioLORD-2023"
    GENERAL_MODEL = "BAAI/bge-small-en-v1.5"
    FALLBACK_MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._medical_model = None
        self._general_model = None
        self._medical_embeddings: dict[str, np.ndarray] = {}
        self._general_embeddings: dict[str, np.ndarray] = {}
        # Fallback: single model if dual load fails
        self._model = None
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._taxonomy_exemplars: dict[str, list[str]] = {}
        self._dual_mode = False
        self._models_loaded: list[str] = []

    @classmethod
    async def get_instance(cls) -> "TextSignalModel":
        """Get or create singleton instance with lazy loading."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self) -> None:
        """Load model and build exemplar embeddings."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        """Load dual models (medical + general) or fall back to single."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Text signal unavailable.")
            return

        self._load_taxonomy_exemplars()
        all_exemplars = {**SPECIALTY_EXEMPLARS, **self._taxonomy_exemplars}

        # Try loading trained model first, then fall back to off-the-shelf
        medical_loaded = False
        general_loaded = False

        # Check for trained model (fine-tuned on medical routing dataset)
        trained_path = Path(__file__).resolve().parent.parent.parent / self.TRAINED_MODEL
        try:
            if trained_path.exists():
                self._medical_model = SentenceTransformer(str(trained_path))
                logger.info("Loaded TRAINED medical embedding: %s", trained_path)
                medical_loaded = True
                self._models_loaded.append(f"trained:{self.TRAINED_MODEL}")
            else:
                raise FileNotFoundError(f"Trained model not found at {trained_path}")
        except Exception as e:
            logger.info("Trained model unavailable (%s), trying BioLORD", e)
            try:
                self._medical_model = SentenceTransformer(self.MEDICAL_MODEL)
                logger.info("Loaded medical embedding: %s", self.MEDICAL_MODEL)
                medical_loaded = True
                self._models_loaded.append(self.MEDICAL_MODEL)
            except Exception as e2:
                logger.warning("BioLORD also unavailable (%s), will use fallback", e2)

        try:
            self._general_model = SentenceTransformer(self.GENERAL_MODEL)
            logger.info("Loaded general embedding: %s", self.GENERAL_MODEL)
            general_loaded = True
            self._models_loaded.append(self.GENERAL_MODEL)
        except Exception as e:
            logger.warning("BGE-small unavailable (%s), will use fallback", e)

        if medical_loaded and general_loaded:
            self._dual_mode = True
            # Pre-compute embeddings for both models
            for specialty, queries in all_exemplars.items():
                # Medical model embeddings
                med_emb = self._medical_model.encode(queries, convert_to_numpy=True)
                med_avg = np.mean(med_emb, axis=0)
                self._medical_embeddings[specialty] = med_avg / np.linalg.norm(med_avg)
                # General model embeddings
                gen_emb = self._general_model.encode(queries, convert_to_numpy=True)
                gen_avg = np.mean(gen_emb, axis=0)
                self._general_embeddings[specialty] = gen_avg / np.linalg.norm(gen_avg)
            logger.info("Dual-embedding mode: %s + %s", self.MEDICAL_MODEL, self.GENERAL_MODEL)
        else:
            # Fallback to single model
            fallback_name = self.MEDICAL_MODEL if medical_loaded else (
                self.GENERAL_MODEL if general_loaded else self.FALLBACK_MODEL
            )
            if not medical_loaded and not general_loaded:
                self._model = SentenceTransformer(self.FALLBACK_MODEL)
                self._models_loaded.append(self.FALLBACK_MODEL)
            elif medical_loaded:
                self._model = self._medical_model
            else:
                self._model = self._general_model

            logger.info("Single-embedding fallback: %s", fallback_name)

            for specialty, queries in all_exemplars.items():
                embeddings = self._model.encode(queries, convert_to_numpy=True)
                avg = np.mean(embeddings, axis=0)
                self._exemplar_embeddings[specialty] = avg / np.linalg.norm(avg)

    def _load_taxonomy_exemplars(self) -> None:
        """Load specialty structure from taxonomy.yaml to generate additional exemplars."""
        taxonomy_path = CONFIG_DIR / "taxonomy.yaml"
        if not taxonomy_path.exists():
            return

        try:
            with open(taxonomy_path) as f:
                taxonomy = yaml.safe_load(f)

            # Add child specialties from taxonomy that aren't already in exemplars
            for domain, specialties in taxonomy.items():
                if not isinstance(specialties, dict):
                    continue
                for specialty, config in specialties.items():
                    key = f"{domain}.{specialty}"
                    if key in SPECIALTY_EXEMPLARS:
                        continue
                    if isinstance(config, dict) and config.get("children"):
                        for child in config["children"]:
                            child_key = f"{key}.{child}"
                            if child_key not in SPECIALTY_EXEMPLARS:
                                # Generate basic exemplars from the name
                                self._taxonomy_exemplars[child_key] = [
                                    f"{child} {specialty} assessment",
                                    f"{child} related {specialty} question",
                                ]
        except Exception as e:
            logger.warning(f"Failed to load taxonomy.yaml: {e}")

    async def match_specialty(self, messages: list[dict]) -> TextSignalResult:
        """Match query text to the best specialty using cosine similarity."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._match_sync, messages)

    def _match_sync(self, messages: list[dict]) -> TextSignalResult:
        """Synchronous specialty matching — dual or single embedding."""
        if not self._dual_mode and self._model is None:
            return TextSignalResult(
                matched_specialty="general.simple_qa",
                similarity=0.0,
                error="Text model not loaded",
            )

        try:
            query_text = self._extract_query_text(messages)
            if not query_text:
                return TextSignalResult(
                    matched_specialty="general.simple_qa",
                    similarity=0.0,
                    error="No text content found",
                )

            if self._dual_mode:
                # Dual-embedding ensemble: max similarity across both models
                med_emb = self._medical_model.encode(query_text, convert_to_numpy=True)
                med_emb = med_emb / np.linalg.norm(med_emb)
                gen_emb = self._general_model.encode(query_text, convert_to_numpy=True)
                gen_emb = gen_emb / np.linalg.norm(gen_emb)

                all_scores: dict[str, float] = {}
                for specialty in set(list(self._medical_embeddings.keys()) + list(self._general_embeddings.keys())):
                    med_score = float(np.dot(med_emb, self._medical_embeddings[specialty])) if specialty in self._medical_embeddings else 0.0
                    gen_score = float(np.dot(gen_emb, self._general_embeddings[specialty])) if specialty in self._general_embeddings else 0.0
                    # Max of both models — medical exemplars score higher on BioLORD,
                    # general exemplars score higher on BGE
                    all_scores[specialty] = max(med_score, gen_score)
            else:
                # Single-model fallback
                query_embedding = self._model.encode(query_text, convert_to_numpy=True)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                all_scores = {}
                for specialty, emb in self._exemplar_embeddings.items():
                    sim = float(np.dot(query_embedding, emb))
                    all_scores[specialty] = sim

            best_specialty = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
            best_score = all_scores[best_specialty]
            is_medical = best_specialty.startswith("medical.")

            return TextSignalResult(
                matched_specialty=best_specialty,
                similarity=best_score,
                all_scores=all_scores,
                is_medical=is_medical,
            )

        except Exception as e:
            logger.error(f"Text signal error: {e}")
            return TextSignalResult(
                matched_specialty="general.simple_qa",
                similarity=0.0,
                error=str(e),
            )

    @staticmethod
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


async def text_signal(messages: list[dict]) -> TextSignalResult:
    """Run the text signal: match query to specialty via embedding similarity.

    Args:
        messages: OpenAI-format message list.

    Returns:
        TextSignalResult with matched_specialty, similarity, and all_scores.
    """
    model = await TextSignalModel.get_instance()
    return await model.match_specialty(messages)
