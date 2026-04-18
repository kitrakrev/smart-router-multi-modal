"""Run all signals in parallel for routing decisions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .complexity import ComplexitySignalResult, complexity_signal
from .modality import ModalitySignalResult, modality_signal
from .safety import SafetySignalResult, safety_signal
from .text import TextSignalResult, text_signal
from .tools import ToolsSignalResult, tools_signal
from .vision import VisionSignalResult, vision_signal

logger = logging.getLogger(__name__)


@dataclass
class AllSignalsResult:
    """Aggregated result from all signal modules."""

    text: TextSignalResult
    vision: VisionSignalResult
    complexity: ComplexitySignalResult
    safety: SafetySignalResult
    tools: ToolsSignalResult
    modality: ModalitySignalResult


async def run_all_signals(
    messages: list[dict], image_data: Optional[str] = None
) -> AllSignalsResult:
    """Run all signal modules in parallel via asyncio.gather.

    Args:
        messages: OpenAI-format message list.
        image_data: Optional raw base64 image string.

    Returns:
        AllSignalsResult with results from all signal modules.
    """
    # Launch all signals concurrently
    text_result, vision_result, complexity_result, safety_result, tools_result, modality_result = (
        await asyncio.gather(
            text_signal(messages),
            vision_signal(messages, image_data),
            complexity_signal(messages),
            safety_signal(messages),
            tools_signal(messages),
            modality_signal(messages, image_data),
            return_exceptions=False,
        )
    )

    return AllSignalsResult(
        text=text_result,
        vision=vision_result,
        complexity=complexity_result,
        safety=safety_result,
        tools=tools_result,
        modality=modality_result,
    )
