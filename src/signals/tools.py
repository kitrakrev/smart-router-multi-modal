"""Detect if query needs medical tools via embedding similarity."""

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


@dataclass
class ToolsSignalResult:
    """Result from tool need detection."""

    needs_tools: bool
    recommended_tools: list[str] = field(default_factory=list)
    tool_scores: dict[str, float] = field(default_factory=dict)
    top_score: float = 0.0
    error: Optional[str] = None


class ToolsSignalModel:
    """Lazy-loaded model for detecting tool needs via embedding match."""

    _instance: Optional["ToolsSignalModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    TOOL_SCORE_THRESHOLD: float = 0.45

    def __init__(self) -> None:
        self._model = None
        self._tool_embeddings: dict[str, np.ndarray] = {}
        self._tool_descriptions: dict[str, str] = {}

    @classmethod
    async def get_instance(cls) -> "ToolsSignalModel":
        """Get or create singleton instance with lazy loading."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self) -> None:
        """Load model and tool config."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous loading."""
        # Load tool config
        tools_config = self._load_tools_config()

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded all-MiniLM-L6-v2 for tools signal")

            # Pre-compute tool embeddings from description + exemplars
            for tool_name, tool_info in tools_config.items():
                description = tool_info.get("description", "")
                exemplars = tool_info.get("exemplars", [])
                self._tool_descriptions[tool_name] = description

                # Combine description and exemplars for richer embedding
                all_texts = [description] + exemplars
                embeddings = self._model.encode(all_texts, convert_to_numpy=True)
                avg = np.mean(embeddings, axis=0)
                avg = avg / np.linalg.norm(avg)
                self._tool_embeddings[tool_name] = avg

        except ImportError:
            logger.error("sentence-transformers not installed. Tools signal unavailable.")

    def _load_tools_config(self) -> dict:
        """Load tool definitions from config/tools.yaml."""
        tools_path = CONFIG_DIR / "tools.yaml"
        if not tools_path.exists():
            logger.warning("tools.yaml not found, using empty tool config")
            return {}

        try:
            with open(tools_path) as f:
                config = yaml.safe_load(f)
            return config.get("tools", {})
        except Exception as e:
            logger.error(f"Failed to load tools.yaml: {e}")
            return {}

    async def detect_tools(self, messages: list[dict]) -> ToolsSignalResult:
        """Detect which medical tools the query might need."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, messages)

    def _detect_sync(self, messages: list[dict]) -> ToolsSignalResult:
        """Synchronous tool detection."""
        if self._model is None or not self._tool_embeddings:
            return ToolsSignalResult(
                needs_tools=False,
                error="Tools model not loaded",
            )

        try:
            query_text = _extract_query_text(messages)
            if not query_text:
                return ToolsSignalResult(needs_tools=False, error="No text content found")

            # Embed query
            query_emb = self._model.encode(query_text, convert_to_numpy=True)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Cosine similarity to each tool
            tool_scores: dict[str, float] = {}
            for tool_name, tool_emb in self._tool_embeddings.items():
                sim = float(np.dot(query_emb, tool_emb))
                tool_scores[tool_name] = sim

            # Determine which tools are needed
            recommended = [
                name
                for name, score in sorted(tool_scores.items(), key=lambda x: -x[1])
                if score >= self.TOOL_SCORE_THRESHOLD
            ]

            top_score = max(tool_scores.values()) if tool_scores else 0.0

            return ToolsSignalResult(
                needs_tools=len(recommended) > 0,
                recommended_tools=recommended,
                tool_scores=tool_scores,
                top_score=top_score,
            )

        except Exception as e:
            logger.error(f"Tools signal error: {e}")
            return ToolsSignalResult(needs_tools=False, error=str(e))


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


async def tools_signal(messages: list[dict]) -> ToolsSignalResult:
    """Run the tools signal: detect if query needs medical tools.

    Args:
        messages: OpenAI-format message list.

    Returns:
        ToolsSignalResult with needs_tools, recommended_tools, and scores.
    """
    model = await ToolsSignalModel.get_instance()
    return await model.detect_tools(messages)
