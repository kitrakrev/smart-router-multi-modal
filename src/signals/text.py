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
    "medical.cardiology": [
        "chest pain with ST elevation on ECG",
        "heart failure management and ejection fraction",
        "atrial fibrillation rate control vs rhythm control",
        "interpret this echocardiogram finding",
    ],
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
    "medical.ophthalmology": [
        "diabetic retinopathy grading from fundus photo",
        "glaucoma assessment with cup-to-disc ratio",
        "macular degeneration wet vs dry classification",
        "retinal detachment signs and management",
    ],
    "medical.emergency": [
        "acute trauma assessment and triage",
        "anaphylaxis emergency management protocol",
        "stroke code rapid evaluation NIHSS",
        "sepsis bundle and hemodynamic resuscitation",
    ],
    "medical.pharmacology": [
        "drug interaction between warfarin and antibiotics",
        "medication dosage adjustment for renal impairment",
        "contraindications for beta blocker therapy",
        "pharmacokinetics of new biologic agents",
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
    """Lazy-loaded MiniLM model for text embedding and specialty matching."""

    _instance: Optional["TextSignalModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model = None
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._taxonomy_exemplars: dict[str, list[str]] = {}

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
        """Synchronous loading of model and exemplar computation."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded all-MiniLM-L6-v2 for text signal")
        except ImportError:
            logger.error("sentence-transformers not installed. Text signal unavailable.")
            return

        # Load taxonomy to enrich exemplars if available
        self._load_taxonomy_exemplars()

        # Merge taxonomy-derived exemplars with defaults
        all_exemplars = {**SPECIALTY_EXEMPLARS, **self._taxonomy_exemplars}

        # Pre-compute averaged embeddings per specialty
        for specialty, queries in all_exemplars.items():
            embeddings = self._model.encode(queries, convert_to_numpy=True)
            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)
            self._exemplar_embeddings[specialty] = avg

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
        """Synchronous specialty matching."""
        if self._model is None:
            return TextSignalResult(
                matched_specialty="general.simple_qa",
                similarity=0.0,
                error="Text model not loaded",
            )

        try:
            # Extract text from the last user message
            query_text = self._extract_query_text(messages)
            if not query_text:
                return TextSignalResult(
                    matched_specialty="general.simple_qa",
                    similarity=0.0,
                    error="No text content found",
                )

            # Embed the query
            query_embedding = self._model.encode(query_text, convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Cosine similarity to each specialty
            all_scores: dict[str, float] = {}
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
