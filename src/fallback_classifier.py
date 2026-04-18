"""Ambiguity fallback classifier using a small LLM.

When the embedding-based router has low confidence (similarity < threshold),
this module calls a small trained LLM to classify the domain.

Uses Qwen2.5-0.5B-Instruct (500M params, ~1GB VRAM, ~200ms inference)
as a fast domain classifier.

Flow:
  1. Router embedding gives similarity < 0.35 → ambiguous
  2. FallbackClassifier.classify(query) → domain label
  3. Router uses the LLM's classification instead of low-confidence embedding

Usage:
  from src.fallback_classifier import fallback_classifier
  result = await fallback_classifier.classify("Is this normal?")
  # result.domain = "general_medicine"
  # result.confidence = 0.85
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

VALID_DOMAINS = [
    "pathology", "radiology", "dermatology", "general_medicine",
    "cardiology", "ophthalmology", "emergency", "pharmacology",
    "code", "reasoning", "creative", "simple_qa",
]

CLASSIFY_PROMPT = """You are a medical query classifier. Classify the user's query into exactly one domain.

Valid domains: pathology, radiology, dermatology, cardiology, ophthalmology, emergency, pharmacology, general_medicine, code, reasoning, creative, simple_qa

Rules:
- If the query mentions tissue, biopsy, cells, histology → pathology
- If the query mentions X-ray, CT, MRI, scan, imaging → radiology
- If the query mentions skin, lesion, rash, mole, dermoscopy → dermatology
- If the query mentions heart, ECG, cardiac → cardiology
- If the query mentions eye, retina, fundus, vision → ophthalmology
- If the query mentions emergency, trauma, urgent, cardiac arrest → emergency
- If the query mentions drug, medication, dose, interaction → pharmacology
- If the query is about code or programming → code
- If the query needs logical analysis → reasoning
- If the query is creative writing → creative
- If it's a simple factual question → simple_qa
- If unclear or could be multiple → general_medicine

Respond with ONLY the domain name, nothing else.

Query: {query}
Domain:"""


@dataclass
class ClassificationResult:
    domain: str
    confidence: float
    source: str = "fallback_llm"
    model_used: str = ""
    latency_ms: float = 0.0


class FallbackClassifier:
    """Small LLM-based domain classifier for ambiguous queries."""

    _instance: Optional["FallbackClassifier"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self._loaded = False
        self._device = "cpu"

    @classmethod
    async def get_instance(cls) -> "FallbackClassifier":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            self._model.eval()
            self._loaded = True
            logger.info("Loaded fallback classifier: %s on %s", self._model_name, self._device)
        except Exception as e:
            logger.warning("Fallback classifier not available: %s", e)

    async def classify(self, query: str) -> ClassificationResult:
        if not self._loaded:
            return ClassificationResult(
                domain="general_medicine", confidence=0.0,
                source="fallback_default", model_used="none",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._classify_sync, query)

    def _classify_sync(self, query: str) -> ClassificationResult:
        import time
        import torch

        t0 = time.perf_counter()
        prompt = CLASSIFY_PROMPT.format(query=query)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True).strip().lower()

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Parse domain from response
        domain = "general_medicine"
        for d in VALID_DOMAINS:
            if d in response:
                domain = d
                break

        # Confidence: 1.0 if exact match, 0.5 if partial
        confidence = 1.0 if response == domain else 0.7

        return ClassificationResult(
            domain=domain,
            confidence=confidence,
            source="fallback_llm",
            model_used=self._model_name,
            latency_ms=round(elapsed_ms, 1),
        )


# Singleton
fallback_classifier = FallbackClassifier()
