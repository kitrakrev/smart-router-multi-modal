"""Vision signal using BioMedCLIP for medical image type classification."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Medical image type exemplar descriptions for cosine matching
IMAGE_TYPE_EXEMPLARS: dict[str, list[str]] = {
    "histology": [
        "histological tissue section stained with H&E",
        "microscopic view of cells and tissue architecture",
        "biopsy specimen under microscope showing cellular morphology",
    ],
    "xray": [
        "X-ray radiograph showing bones and soft tissue",
        "chest X-ray with lung fields and cardiac silhouette",
        "plain film radiograph of skeletal structures",
    ],
    "ct": [
        "CT scan axial slice showing cross-sectional anatomy",
        "computed tomography image with contrast enhancement",
        "CT reconstruction of abdominal or thoracic structures",
    ],
    "mri": [
        "MRI scan showing soft tissue contrast",
        "magnetic resonance image T1 or T2 weighted sequence",
        "brain MRI showing white and gray matter differentiation",
    ],
    "dermoscopy": [
        "dermoscopic image of skin lesion with pigment network",
        "dermatoscopy showing skin surface structures and patterns",
        "close-up dermoscopic view of mole or nevus",
    ],
    "fundus": [
        "retinal fundus photograph showing optic disc and macula",
        "fundoscopy image of retinal vasculature",
        "color fundus photo showing retinal pathology",
    ],
    "clinical_photo": [
        "clinical photograph of patient presentation",
        "medical photography of external anatomical finding",
        "clinical image showing visible signs and symptoms",
    ],
    "ecg": [
        "electrocardiogram tracing with leads and waveforms",
        "ECG strip showing P waves QRS complexes and T waves",
        "12-lead ECG recording of cardiac electrical activity",
    ],
}


@dataclass
class VisionSignalResult:
    """Result from vision signal analysis."""

    image_type: str
    similarity_score: float
    all_scores: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class VisionSignalModel:
    """Lazy-loaded BioMedCLIP model for medical image classification."""

    _instance: Optional["VisionSignalModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._device: str = "cpu"
        self._model_name: str = ""

    @classmethod
    async def get_instance(cls) -> "VisionSignalModel":
        """Get or create singleton instance with lazy loading."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load_model()
        return cls._instance

    async def _load_model(self) -> None:
        """Load BioMedCLIP or fallback to openai/clip."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading with fallback."""
        try:
            import open_clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try BioMedCLIP first
            try:
                model, _, processor = open_clip.create_model_and_transforms(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                tokenizer = open_clip.get_tokenizer(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                self._model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                logger.info("Loaded BioMedCLIP model")
            except Exception as e:
                logger.warning(f"BioMedCLIP unavailable ({e}), falling back to openai/clip")
                model, _, processor = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._model_name = "openai/clip-vit-base-patch32"

            model = model.to(self._device).eval()
            self._model = model
            self._processor = processor
            self._tokenizer = tokenizer

            # Pre-compute exemplar embeddings
            self._precompute_exemplar_embeddings()

        except ImportError:
            logger.warning(
                "open_clip not installed. Vision signal will use sentence-transformers fallback."
            )
            self._load_sentence_transformer_fallback()

    def _load_sentence_transformer_fallback(self) -> None:
        """Fallback using sentence-transformers CLIP model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("clip-ViT-B-32")
            self._model_name = "clip-ViT-B-32-sentence-transformers"
            self._precompute_exemplar_embeddings_st()
            logger.info("Loaded sentence-transformers CLIP fallback")
        except ImportError:
            logger.error("No vision model available. Install open_clip or sentence-transformers.")

    def _precompute_exemplar_embeddings(self) -> None:
        """Pre-compute text embeddings for all image type exemplars using open_clip."""
        import torch

        for image_type, descriptions in IMAGE_TYPE_EXEMPLARS.items():
            tokens = self._tokenizer(descriptions).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Average the exemplar embeddings for this type
                avg_embedding = text_features.mean(dim=0)
                avg_embedding = avg_embedding / avg_embedding.norm()
                self._exemplar_embeddings[image_type] = avg_embedding.cpu().numpy()

    def _precompute_exemplar_embeddings_st(self) -> None:
        """Pre-compute exemplar embeddings using sentence-transformers."""
        for image_type, descriptions in IMAGE_TYPE_EXEMPLARS.items():
            embeddings = self._model.encode(descriptions, convert_to_numpy=True)
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            self._exemplar_embeddings[image_type] = avg_embedding

    def _decode_base64_image(self, image_data: str) -> "PIL.Image.Image":
        """Decode base64 image string to PIL Image."""
        from PIL import Image

        # Handle data URI format: data:image/png;base64,<data>
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _extract_image_from_messages(
        self, messages: list[dict], image_data: Optional[str] = None
    ) -> Optional["PIL.Image.Image"]:
        """Extract image from OpenAI multimodal message format or raw base64."""
        from PIL import Image

        # Direct base64 input
        if image_data:
            return self._decode_base64_image(image_data)

        # Search messages for image_url content blocks
        for message in reversed(messages):
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            return self._decode_base64_image(url)
        return None

    async def classify_image(
        self, messages: list[dict], image_data: Optional[str] = None
    ) -> VisionSignalResult:
        """Classify a medical image by type using cosine similarity to exemplars."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._classify_image_sync, messages, image_data
        )

    def _classify_image_sync(
        self, messages: list[dict], image_data: Optional[str] = None
    ) -> VisionSignalResult:
        """Synchronous image classification."""
        try:
            image = self._extract_image_from_messages(messages, image_data)
            if image is None:
                return VisionSignalResult(
                    image_type="none",
                    similarity_score=0.0,
                    error="No image found in input",
                )

            # Get image embedding
            image_embedding = self._embed_image(image)
            if image_embedding is None:
                return VisionSignalResult(
                    image_type="unknown",
                    similarity_score=0.0,
                    error="Failed to embed image",
                )

            # Cosine similarity to each image type
            all_scores: dict[str, float] = {}
            for image_type, exemplar_emb in self._exemplar_embeddings.items():
                sim = float(np.dot(image_embedding, exemplar_emb))
                all_scores[image_type] = sim

            # Best match
            best_type = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
            best_score = all_scores[best_type]

            return VisionSignalResult(
                image_type=best_type,
                similarity_score=best_score,
                all_scores=all_scores,
            )

        except Exception as e:
            logger.error(f"Vision signal error: {e}")
            return VisionSignalResult(
                image_type="unknown",
                similarity_score=0.0,
                error=str(e),
            )

    def _embed_image(self, image: "PIL.Image.Image") -> Optional[np.ndarray]:
        """Embed a PIL image to a normalized vector."""
        if self._model is None:
            return None

        try:
            # sentence-transformers fallback
            if self._model_name.endswith("sentence-transformers"):
                embedding = self._model.encode(image, convert_to_numpy=True)
                return embedding / np.linalg.norm(embedding)

            # open_clip path
            import torch

            image_input = self._processor(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return None


async def vision_signal(
    messages: list[dict], image_data: Optional[str] = None
) -> VisionSignalResult:
    """Run the vision signal: classify medical image type.

    Args:
        messages: OpenAI-format message list (may contain image_url blocks).
        image_data: Optional raw base64 image string.

    Returns:
        VisionSignalResult with image_type, similarity_score, and all_scores.
    """
    model = await VisionSignalModel.get_instance()
    return await model.classify_image(messages, image_data)
