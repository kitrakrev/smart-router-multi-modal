"""Prompt management: per-model x per-specialty templates."""
from src.prompts.manager import PromptManager, prompt_manager
from src.prompts.templates import TemplateEngine, template_engine

__all__ = [
    "PromptManager",
    "prompt_manager",
    "TemplateEngine",
    "template_engine",
]
