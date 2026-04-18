"""Per-model x per-specialty prompt manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from src.prompts.templates import TemplateEngine, template_engine

logger = logging.getLogger(__name__)


class PromptResult:
    """Result of a prompt lookup."""

    def __init__(
        self,
        system_prompt: str,
        params: dict[str, Any],
        source: str,  # "auto" | "manual" | "dspy"
        model_name: str = "",
        specialty: str = "",
    ):
        self.system_prompt = system_prompt
        self.params = params
        self.source = source
        self.model_name = model_name
        self.specialty = specialty

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "params": self.params,
            "source": self.source,
            "model_name": self.model_name,
            "specialty": self.specialty,
        }


class PromptManager:
    """Manage prompts with priority: DSPy override > manual override > auto-generated."""

    def __init__(self, engine: Optional[TemplateEngine] = None) -> None:
        self._engine = engine or template_engine
        self._manual_overrides: dict[str, dict[str, Any]] = {}
        # key: "model_name::specialty" → {"system_prompt": ..., "params": ...}
        self._dspy_overrides: dict[str, dict[str, Any]] = {}
        self._taxonomy: dict[str, Any] = {}  # parent-child specialty map

    def load(self, config_path: str | Path) -> None:
        """Load prompt_templates.yaml and any overrides."""
        self._engine.load(config_path)

        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}

            # Load DSPy overrides section
            overrides = cfg.get("overrides", {})
            if isinstance(overrides, dict):
                for key, val in overrides.items():
                    if isinstance(val, dict):
                        self._dspy_overrides[key] = val

    def load_taxonomy(self, taxonomy_path: str | Path) -> None:
        """Load taxonomy.yaml for specialty inheritance."""
        path = Path(taxonomy_path)
        if not path.exists():
            return

        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

        self._taxonomy = cfg

    def set_manual_override(
        self, model_name: str, specialty: str, system_prompt: str, params: Optional[dict] = None
    ) -> None:
        """Set a manual prompt override for a model+specialty pair."""
        key = f"{model_name}::{specialty}"
        self._manual_overrides[key] = {
            "system_prompt": system_prompt,
            "params": params or {},
        }

    def get_prompt(
        self,
        model_name: str,
        model_type: str,
        specialty: str,
    ) -> PromptResult:
        """Look up the best prompt for a model+specialty combination.

        Priority:
          1. DSPy override (if exists)
          2. Manual override
          3. Auto-generated from template engine
        """
        key = f"{model_name}::{specialty}"

        # 1. DSPy override
        if key in self._dspy_overrides:
            ov = self._dspy_overrides[key]
            params = self._engine.get_params(model_type)
            params.update(ov.get("params", {}))
            return PromptResult(
                system_prompt=ov.get("system_prompt", ""),
                params=params,
                source="dspy",
                model_name=model_name,
                specialty=specialty,
            )

        # 2. Manual override
        if key in self._manual_overrides:
            ov = self._manual_overrides[key]
            params = self._engine.get_params(model_type)
            params.update(ov.get("params", {}))
            return PromptResult(
                system_prompt=ov.get("system_prompt", ""),
                params=params,
                source="manual",
                model_name=model_name,
                specialty=specialty,
            )

        # 3. Auto-generated
        # Handle inheritance: if child specialty has no task_instruction, use parent's
        spec_info = self._engine.get_specialty_info(specialty)
        if not spec_info.get("task_instruction"):
            parent = self._find_parent_specialty(specialty)
            if parent:
                specialty = parent

        system_prompt = self._engine.format_prompt(model_type, specialty)
        params = self._engine.get_params(model_type)

        return PromptResult(
            system_prompt=system_prompt,
            params=params,
            source="auto",
            model_name=model_name,
            specialty=specialty,
        )

    def _find_parent_specialty(self, child: str) -> Optional[str]:
        """Find the parent specialty for a child in the taxonomy."""
        for _domain, specialties in self._taxonomy.items():
            if not isinstance(specialties, dict):
                continue
            for parent_name, parent_info in specialties.items():
                if isinstance(parent_info, dict):
                    children = parent_info.get("children", [])
                    if child in children:
                        return parent_name
        return None

    def get_prompt_matrix(self) -> list[dict[str, Any]]:
        """Return the full model x specialty prompt matrix for the dashboard."""
        matrix = []
        specialties = self._engine.list_specialties()

        for spec in specialties:
            name = spec["name"]
            for mt in self._engine.list_model_types():
                key_dspy = f"{mt}::{name}"
                key_manual = f"{mt}::{name}"

                if key_dspy in self._dspy_overrides:
                    source = "dspy"
                    prompt = self._dspy_overrides[key_dspy].get("system_prompt", "")
                elif key_manual in self._manual_overrides:
                    source = "manual"
                    prompt = self._manual_overrides[key_manual].get("system_prompt", "")
                else:
                    source = "auto"
                    prompt = self._engine.format_prompt(mt, name)

                matrix.append({
                    "model_type": mt,
                    "specialty": name,
                    "source": source,
                    "prompt_preview": prompt[:50],
                    "full_prompt": prompt,
                    "params": self._engine.get_params(mt),
                })

        return matrix


# Singleton
prompt_manager = PromptManager()
