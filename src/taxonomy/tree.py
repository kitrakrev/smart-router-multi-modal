"""Hierarchical specialty taxonomy with config inheritance and embedding match."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from .aliases import resolve_alias, resolve_alias_fuzzy

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class SpecialtyNode:
    """A node in the specialty taxonomy tree."""

    name: str
    path: str  # e.g., "medical.radiology.chest"
    children: list["SpecialtyNode"] = field(default_factory=list)
    image_types: list[str] = field(default_factory=list)
    parent: Optional["SpecialtyNode"] = None
    config: dict[str, Any] = field(default_factory=dict)

    def get_child(self, name: str) -> Optional["SpecialtyNode"]:
        """Find a direct child by name."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def get_all_image_types(self) -> list[str]:
        """Get image types from this node and all children."""
        types = list(self.image_types)
        for child in self.children:
            types.extend(child.get_all_image_types())
        return list(set(types))

    def get_inherited_config(self) -> dict[str, Any]:
        """Get config with parent inheritance (child overrides parent)."""
        if self.parent is None:
            return dict(self.config)
        parent_config = self.parent.get_inherited_config()
        parent_config.update(self.config)
        return parent_config

    def flatten(self) -> list["SpecialtyNode"]:
        """Return this node and all descendants as a flat list."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def __repr__(self) -> str:
        child_names = [c.name for c in self.children]
        return f"SpecialtyNode({self.path}, children={child_names})"


class SpecialtyTree:
    """Hierarchical specialty taxonomy loaded from config/taxonomy.yaml."""

    _instance: Optional["SpecialtyTree"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self.roots: dict[str, SpecialtyNode] = {}
        self._all_nodes: dict[str, SpecialtyNode] = {}
        self._model = None
        self._node_embeddings: dict[str, np.ndarray] = {}

    @classmethod
    async def get_instance(cls) -> "SpecialtyTree":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._load_from_yaml()
                    await cls._instance._build_embeddings()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> "SpecialtyTree":
        """Synchronous access (tree only, no embeddings)."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_from_yaml()
        return cls._instance

    def _load_from_yaml(self) -> None:
        """Load taxonomy structure from config/taxonomy.yaml."""
        taxonomy_path = CONFIG_DIR / "taxonomy.yaml"
        if not taxonomy_path.exists():
            logger.error(f"taxonomy.yaml not found at {taxonomy_path}")
            return

        try:
            with open(taxonomy_path) as f:
                taxonomy = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load taxonomy.yaml: {e}")
            return

        # Build tree from YAML
        for domain_name, specialties in taxonomy.items():
            domain_node = SpecialtyNode(name=domain_name, path=domain_name)
            self.roots[domain_name] = domain_node
            self._all_nodes[domain_name] = domain_node

            if not isinstance(specialties, dict):
                continue

            for specialty_name, spec_config in specialties.items():
                spec_path = f"{domain_name}.{specialty_name}"
                spec_node = SpecialtyNode(
                    name=specialty_name,
                    path=spec_path,
                    parent=domain_node,
                )

                if isinstance(spec_config, dict):
                    spec_node.image_types = spec_config.get("image_types", [])
                    spec_node.config = {
                        k: v
                        for k, v in spec_config.items()
                        if k not in ("children", "image_types")
                    }

                    # Build child nodes
                    for child_name in spec_config.get("children", []):
                        child_path = f"{spec_path}.{child_name}"
                        child_node = SpecialtyNode(
                            name=child_name,
                            path=child_path,
                            parent=spec_node,
                        )
                        spec_node.children.append(child_node)
                        self._all_nodes[child_path] = child_node

                domain_node.children.append(spec_node)
                self._all_nodes[spec_path] = spec_node

        logger.info(f"Loaded taxonomy with {len(self._all_nodes)} nodes")

    async def _build_embeddings(self) -> None:
        """Build node name embeddings for fuzzy matching."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._build_embeddings_sync)

    def _build_embeddings_sync(self) -> None:
        """Synchronous embedding computation for all taxonomy nodes."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")

            for path, node in self._all_nodes.items():
                # Create description from path components
                components = path.split(".")
                description = " ".join(components)
                emb = self._model.encode(description, convert_to_numpy=True)
                emb = emb / np.linalg.norm(emb)
                self._node_embeddings[path] = emb

        except ImportError:
            logger.warning("sentence-transformers not installed. Taxonomy embedding unavailable.")

    def get_node(self, path: str) -> Optional[SpecialtyNode]:
        """Get a node by its full path (e.g., 'medical.radiology.chest')."""
        return self._all_nodes.get(path)

    def resolve(self, term: str) -> Optional[SpecialtyNode]:
        """Resolve a term to a taxonomy node using alias lookup.

        Args:
            term: A specialty name, alias, or path.

        Returns:
            SpecialtyNode or None.
        """
        # Direct path match
        if term in self._all_nodes:
            return self._all_nodes[term]

        # Alias resolution
        canonical = resolve_alias(term)
        if canonical:
            # Try with medical. prefix
            for prefix in ["medical.", "general.", ""]:
                full_path = f"{prefix}{canonical}" if prefix else canonical
                if full_path in self._all_nodes:
                    return self._all_nodes[full_path]

        return None

    async def resolve_fuzzy(self, term: str, threshold: float = 0.4) -> Optional[SpecialtyNode]:
        """Resolve a term using alias lookup + embedding fuzzy match.

        Args:
            term: A specialty name or free-text query.
            threshold: Minimum cosine similarity for fuzzy match.

        Returns:
            SpecialtyNode or None.
        """
        # Try exact resolution first
        node = self.resolve(term)
        if node is not None:
            return node

        # Try fuzzy alias resolution
        canonical = await resolve_alias_fuzzy(term, threshold)
        if canonical:
            for prefix in ["medical.", "general.", ""]:
                full_path = f"{prefix}{canonical}" if prefix else canonical
                if full_path in self._all_nodes:
                    return self._all_nodes[full_path]

        # Embedding match against taxonomy nodes
        if self._model is not None and self._node_embeddings:
            term_emb = self._model.encode(term, convert_to_numpy=True)
            term_emb = term_emb / np.linalg.norm(term_emb)

            best_path: Optional[str] = None
            best_score = threshold

            for path, emb in self._node_embeddings.items():
                sim = float(np.dot(term_emb, emb))
                if sim > best_score:
                    best_score = sim
                    best_path = path

            if best_path:
                return self._all_nodes[best_path]

        return None

    def get_specialties_for_image_type(self, image_type: str) -> list[SpecialtyNode]:
        """Find all specialty nodes that handle a given image type."""
        results: list[SpecialtyNode] = []
        for path, node in self._all_nodes.items():
            if image_type in node.image_types:
                results.append(node)
        return results

    def get_medical_specialties(self) -> list[SpecialtyNode]:
        """Get all medical specialty nodes (direct children of 'medical')."""
        medical = self.roots.get("medical")
        if medical is None:
            return []
        return list(medical.children)

    def get_all_nodes(self) -> dict[str, SpecialtyNode]:
        """Get all nodes indexed by path."""
        return dict(self._all_nodes)
