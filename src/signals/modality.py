"""Detect input modality: text_only, vision, or multimodal."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModalitySignalResult:
    """Result from modality detection."""

    modality: str  # "text_only", "vision", "multimodal"
    has_image: bool
    has_text: bool
    image_count: int = 0
    error: Optional[str] = None


async def modality_signal(
    messages: list[dict], image_data: Optional[str] = None
) -> ModalitySignalResult:
    """Detect the modality of the input request.

    Checks for:
    - image_url content blocks in OpenAI multimodal format
    - Direct base64 image data
    - Text content

    Args:
        messages: OpenAI-format message list.
        image_data: Optional raw base64 image string.

    Returns:
        ModalitySignalResult with modality type.
    """
    has_image = False
    has_text = False
    image_count = 0

    # Check for direct image data
    if image_data:
        has_image = True
        image_count += 1

    # Scan messages for image_url blocks and text
    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            if content.strip():
                has_text = True

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if isinstance(block, str) and block.strip():
                        has_text = True
                    continue

                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    if text.strip():
                        has_text = True

                elif block_type == "image_url":
                    has_image = True
                    image_count += 1

                elif block_type == "image":
                    has_image = True
                    image_count += 1

    # Determine modality
    if has_image and has_text:
        modality = "multimodal"
    elif has_image:
        modality = "vision"
    else:
        modality = "text_only"

    return ModalitySignalResult(
        modality=modality,
        has_image=has_image,
        has_text=has_text,
        image_count=image_count,
    )
