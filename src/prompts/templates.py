"""Generate prompts from model_type x specialty templates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Load model_types and specialties from prompt_templates.yaml and generate prompts."""

    def __init__(self) -> None:
        self._model_types: dict[str, Any] = {}
        self._specialties: dict[str, Any] = {}
        self._loaded = False

    def load(self, config_path: str | Path) -> None:
        """Load templates from prompt_templates.yaml."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Prompt templates config not found: %s", path)
            return

        with open(path) as f:
            cfg = yaml.safe_load(f)

        self._model_types = cfg.get("model_types", {})
        self._specialties = cfg.get("specialties", {})
        self._loaded = True
        logger.info(
            "Loaded %d model types, %d specialties",
            len(self._model_types),
            len(self._specialties),
        )

    def get_specialty_info(self, specialty_name: str) -> dict[str, str]:
        """Get title and task_instruction for a specialty."""
        info = self._specialties.get(specialty_name, {})
        return {
            "title": info.get("title", specialty_name.replace("_", " ")),
            "task_instruction": info.get("task_instruction", ""),
        }

    def format_prompt(self, model_type: str, specialty_name: str) -> str:
        """Generate a system prompt from model_type x specialty.

        Uses the template for the model_type, filled with specialty title
        and task_instruction.
        """
        mt = self._model_types.get(model_type)
        if mt is None:
            # Fallback to generalist
            mt = self._model_types.get("generalist", {"template": "You are a {specialty_title}. {task_instruction}"})

        template = mt.get("template", "You are a {specialty_title}. {task_instruction}")
        spec = self.get_specialty_info(specialty_name)

        return template.format(
            specialty_title=spec["title"],
            task_instruction=spec["task_instruction"],
        ).strip()

    def get_params(self, model_type: str, specialty_name: str | None = None) -> dict[str, Any]:
        """Get generation parameters for a model_type.

        Returns dict with: temperature, reasoning, thinking_tokens, max_tokens.
        """
        mt = self._model_types.get(model_type)
        if mt is None:
            mt = self._model_types.get("generalist", {})

        params = dict(mt.get("params", {}))

        # Defaults
        params.setdefault("temperature", 0.3)
        params.setdefault("max_tokens", 1024)

        return params

    def list_model_types(self) -> list[str]:
        return list(self._model_types.keys())

    def list_specialties(self) -> list[dict[str, Any]]:
        return [
            {"name": name, **info}
            for name, info in self._specialties.items()
        ]


# Singleton
template_engine = TemplateEngine()
