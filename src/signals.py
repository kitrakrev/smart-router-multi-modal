"""Signal extractors for the LLM Router.

Each signal analyzes the incoming query and returns a SignalResult
with a score (0-1), confidence, execution time, and metadata.
All signals run in parallel via asyncio.gather for minimum latency.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    name: str
    score: float
    confidence: float
    execution_time_ms: float
    metadata: dict = field(default_factory=dict)
    skipped: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed(fn):
    """Decorator that injects execution_time_ms into the returned SignalResult."""
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result: SignalResult = await fn(*args, **kwargs)
        result.execution_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result
    return wrapper


def _extract_text(messages: list[dict]) -> str:
    """Pull all user text from the messages array."""
    parts = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
    return " ".join(parts)


def _has_image(messages: list[dict]) -> bool:
    """Check if any message contains an image_url block."""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


def _extract_image_metadata(messages: list[dict]) -> dict:
    """Extract image type and approximate size from the first image_url block."""
    import base64 as _b64
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    image_type = "unknown"
                    size_kb = 0
                    if url.startswith("data:"):
                        # data:image/png;base64,AAAA...
                        header, _, data = url.partition(",")
                        mime = header.split(";")[0].replace("data:", "")
                        image_type = mime  # e.g. image/png
                        try:
                            raw = _b64.b64decode(data)
                            size_kb = round(len(raw) / 1024, 1)
                        except Exception:
                            size_kb = round(len(data) * 3 / 4 / 1024, 1)
                    elif url.startswith("http"):
                        image_type = "url"
                    return {"image_type": image_type, "image_size_kb": size_kb}
    return {}


# ---------------------------------------------------------------------------
# 1. KeywordSignal
# ---------------------------------------------------------------------------

_KEYWORD_PATTERNS: dict[str, list[str]] = {
    "math": [r"\bsolve\b", r"\bcalculat[e|ion]\b", r"\bintegral\b", r"\bderivative\b",
             r"\bequation\b", r"\bproof\b", r"\balgebra\b", r"\bmatrix\b", r"\bstatistic"],
    "code": [r"\bcode\b", r"\bfunction\b", r"\bpython\b", r"\bjavascript\b", r"\brust\b",
             r"\bbug\b", r"\bapi\b", r"\bclass\b", r"\bimport\b", r"\bcompile\b",
             r"\brefactor\b", r"\bimplementation\b", r"\balgorithm\b"],
    "creative": [r"\bwrite\s+(a\s+)?(poem|story|essay|song|haiku)\b", r"\bcreative\b",
                 r"\bimagine\b", r"\bfiction\b"],
    "medical": [r"\bsymptom\b", r"\bdiagnos[ei]s\b", r"\btreat(ment)?\b", r"\bmedicine\b",
                r"\bdosage\b", r"\bpatient\b"],
    "legal": [r"\blegal\b", r"\bcontract\b", r"\bclause\b", r"\bstatute\b", r"\bliability\b"],
    "science": [r"\bquantum\b", r"\bmolecul\b", r"\breaction\b", r"\bphysics\b",
                r"\bchemistry\b", r"\bbiology\b", r"\bexperiment\b"],
}

@_timed
async def keyword_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages).lower()
    scores: dict[str, float] = {}
    for domain, patterns in _KEYWORD_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, text))
        scores[domain] = min(hits / max(len(patterns) * 0.4, 1), 1.0)
    best_domain = max(scores, key=scores.get) if scores else "general"
    best_score = scores.get(best_domain, 0.0)
    # If all scores are 0, default to "general"
    if best_score == 0.0:
        best_domain = "general"
    return SignalResult(
        name="keyword",
        score=best_score,
        confidence=min(best_score + 0.2, 1.0),
        execution_time_ms=0,
        metadata={"domain_hint": best_domain, "all_scores": scores},
    )


# ---------------------------------------------------------------------------
# 2. DomainSignal  (embedding-based with lazy-loaded MiniLM)
# ---------------------------------------------------------------------------

_DOMAIN_EXEMPLARS: dict[str, list[str]] = {
    "math": ["Solve the integral of x^2 dx", "Prove that the sum of angles in a triangle is 180",
             "Find the eigenvalues of a matrix", "What is the derivative of sin(x)?"],
    "code": ["Write a Python function to sort a list", "Fix this JavaScript bug",
             "Implement a binary search tree in Rust", "Explain Big-O notation"],
    "science": ["Explain quantum entanglement", "What happens in a chemical reaction?",
                "How does photosynthesis work?", "Describe the structure of DNA"],
    "creative": ["Write a poem about autumn", "Tell me a short story about a robot",
                 "Compose a haiku about the ocean"],
    "medical": ["What are symptoms of diabetes?", "How is hypertension treated?",
                "Explain the mechanism of aspirin"],
    "legal": ["What is a non-compete clause?", "Explain tort liability",
              "Summarize the GDPR requirements"],
    "general": ["What is the capital of France?", "How do I cook rice?",
                "Tell me a joke", "What time is it?"],
}

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is REQUIRED for domain_signal embeddings. "
        "Install it: pip install sentence-transformers"
    )

import numpy as np

_domain_model: SentenceTransformer | None = None
_domain_embeddings: dict[str, Any] = {}


def _get_domain_model() -> SentenceTransformer:
    """Lazy singleton: load MiniLM model once, reuse on every call."""
    global _domain_model, _domain_embeddings
    if _domain_model is not None:
        return _domain_model
    _domain_model = SentenceTransformer("all-MiniLM-L6-v2")
    for domain, exemplars in _DOMAIN_EXEMPLARS.items():
        embs = _domain_model.encode(exemplars)
        _domain_embeddings[domain] = np.mean(embs, axis=0)
    return _domain_model


@_timed
async def domain_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages)
    model = _get_domain_model()
    query_emb = model.encode([text])[0]
    scores = {}
    for domain, centroid in _domain_embeddings.items():
        cos_sim = float(np.dot(query_emb, centroid) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(centroid) + 1e-8))
        scores[domain] = max(0.0, cos_sim)
    best = max(scores, key=scores.get)
    return SignalResult(
        name="domain", score=scores[best], confidence=0.85,
        execution_time_ms=0,
        metadata={"domain": best, "all_scores": {k: round(v, 3) for k, v in scores.items()}},
    )


# ---------------------------------------------------------------------------
# 3. ComplexitySignal
# ---------------------------------------------------------------------------

_QUESTION_WORDS = {"what", "why", "how", "explain", "describe", "compare", "analyze",
                   "evaluate", "derive", "prove", "justify"}
_TECHNICAL_TERMS = {"algorithm", "optimization", "architecture", "distributed", "concurrent",
                    "asymptotic", "polynomial", "differential", "stochastic", "recurrence",
                    "eigenvalue", "topology", "morphism", "isomorphism", "homogeneous",
                    "heterogeneous", "bayesian", "regression", "gradient", "backpropagation",
                    "integral", "derivative", "theorem", "epsilon", "delta", "proof",
                    "convergence", "divergence", "riemann", "fourier", "laplace",
                    "kubernetes", "microservice", "terraform", "cryptography", "recursion",
                    "complexity", "heuristic", "deterministic", "nondeterministic"}

@_timed
async def complexity_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages)
    words = text.lower().split()
    word_count = len(words)

    # Factors
    length_score = min(word_count / 200, 1.0)
    question_count = sum(1 for w in words if w.strip("?.,!") in _QUESTION_WORDS)
    question_score = min(question_count / 3, 1.0)
    tech_count = sum(1 for w in words if w.strip("?.,!") in _TECHNICAL_TERMS)
    tech_score = min(tech_count / 3, 1.0)
    sentence_count = max(text.count(".") + text.count("?") + text.count("!"), 1)
    multi_step = min(sentence_count / 5, 1.0)

    # Tech density bonus: high tech term density in short queries = complex
    density = tech_count / max(word_count, 1)
    density_bonus = min(density * 5, 0.3)

    score = 0.15 * length_score + 0.2 * question_score + 0.35 * tech_score + 0.15 * multi_step + 0.15 * density_bonus
    score = round(min(score, 1.0), 3)

    return SignalResult(
        name="complexity", score=score, confidence=0.7,
        execution_time_ms=0,
        metadata={"word_count": word_count, "question_words": question_count,
                  "technical_terms": tech_count, "sentences": sentence_count},
    )


# ---------------------------------------------------------------------------
# 4. LanguageSignal
# ---------------------------------------------------------------------------

@_timed
async def language_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages)
    try:
        from langdetect import detect, detect_langs
        lang = detect(text)
        probs = detect_langs(text)
        confidence = probs[0].prob if probs else 0.5
        is_english = 1.0 if lang == "en" else 0.0
        return SignalResult(
            name="language", score=is_english, confidence=round(confidence, 2),
            execution_time_ms=0,
            metadata={"detected": lang, "probabilities": {str(p.lang): round(p.prob, 3) for p in probs[:5]}},
        )
    except Exception:
        return SignalResult(
            name="language", score=1.0, confidence=0.3,
            execution_time_ms=0,
            metadata={"detected": "en", "fallback": True},
        )


# ---------------------------------------------------------------------------
# 5. SafetySignal
# ---------------------------------------------------------------------------

_SAFETY_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"pretend\s+you\s+are",
    r"you\s+are\s+now\s+(DAN|jailbroken|evil)",
    r"bypass\s+(safety|filter|restriction)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(rules|restrictions|guidelines)",
    r"do\s+anything\s+now",
    r"(hack|exploit|attack|ddos|phishing|malware)\s+(a|the|this|my)",
    r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|poison|virus)",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",
]

@_timed
async def safety_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages).lower()
    hits = []
    for pattern in _SAFETY_PATTERNS:
        if re.search(pattern, text):
            hits.append(pattern)
    score = min(len(hits) / 2, 1.0)
    return SignalResult(
        name="safety", score=round(score, 3),
        confidence=0.8 if hits else 0.9,
        execution_time_ms=0,
        metadata={"matched_patterns": len(hits), "flagged": score > 0.5},
    )


# ---------------------------------------------------------------------------
# 6. PIISignal
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

@_timed
async def pii_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages)
    found: dict[str, int] = {}
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = len(matches)
    total = sum(found.values())
    score = min(total / 3, 1.0)
    return SignalResult(
        name="pii", score=round(score, 3), confidence=0.9,
        execution_time_ms=0,
        metadata={"detected_types": found, "total_matches": total},
    )


# ---------------------------------------------------------------------------
# 7. VisionSignal
# ---------------------------------------------------------------------------

@_timed
async def vision_signal(messages: list[dict], **_kw) -> SignalResult:
    has_img = _has_image(messages)
    if not has_img:
        return SignalResult(
            name="vision", score=0.0, confidence=1.0,
            execution_time_ms=0, metadata={"detected": False}, skipped=True,
        )
    # If image present, classify content type from text context
    text = _extract_text(messages).lower()
    content_type = "photo"
    if any(w in text for w in ["chart", "graph", "plot", "data"]):
        content_type = "chart"
    elif any(w in text for w in ["diagram", "flow", "architecture", "uml"]):
        content_type = "diagram"
    elif any(w in text for w in ["screenshot", "screen", "ui", "app", "page"]):
        content_type = "screenshot"
    elif any(w in text for w in ["code", "terminal", "console", "output"]):
        content_type = "code_screenshot"

    # Extract image metadata (type, size)
    img_meta = _extract_image_metadata(messages)

    return SignalResult(
        name="vision", score=1.0, confidence=0.7,
        execution_time_ms=0,
        metadata={"detected": True, "content_type": content_type, **img_meta},
    )


# ---------------------------------------------------------------------------
# 8. ToolSignal
# ---------------------------------------------------------------------------

_TOOL_KEYWORDS = [
    "search", "look up", "find", "calculate", "compute", "scan", "block",
    "firewall", "vulnerability", "quarantine", "report", "cve", "port scan",
    "ip reputation", "run", "execute", "check"
]

@_timed
async def tool_signal(messages: list[dict], tools: Optional[list] = None, **_kw) -> SignalResult:
    text = _extract_text(messages).lower()
    # If tools are explicitly provided
    if tools:
        return SignalResult(
            name="tool", score=1.0, confidence=1.0,
            execution_time_ms=0,
            metadata={"needed": True, "reason": "tools_provided", "tool_count": len(tools)},
        )
    # Keyword detection
    hits = sum(1 for kw in _TOOL_KEYWORDS if kw in text)
    score = min(hits / 3, 1.0)
    return SignalResult(
        name="tool", score=round(score, 3), confidence=0.6,
        execution_time_ms=0,
        metadata={"needed": score > 0.5, "keyword_hits": hits},
    )


# ---------------------------------------------------------------------------
# 9. ModalitySignal
# ---------------------------------------------------------------------------

_IMAGE_GEN_PATTERNS = [
    r"\b(generate|create|draw|make)\s+(an?\s+)?(image|picture|illustration|photo|art)",
    r"\b(dall-?e|midjourney|stable\s*diffusion)\b",
    r"\bimage\s+gen",
]

_CODE_GEN_PATTERNS = [
    r"\b(write|generate|create|implement|build|code)\s+(a\s+)?(function|class|script|program|app|api|server)",
    r"\b(refactor|debug|fix)\s+(this|the|my)\s+(code|function|script)",
]

@_timed
async def modality_signal(messages: list[dict], **_kw) -> SignalResult:
    text = _extract_text(messages).lower()
    has_img = _has_image(messages)

    if has_img:
        modality = "vision"
    elif any(re.search(p, text) for p in _IMAGE_GEN_PATTERNS):
        modality = "image_gen"
    elif any(re.search(p, text) for p in _CODE_GEN_PATTERNS):
        modality = "code_gen"
    else:
        modality = "text"

    return SignalResult(
        name="modality", score=1.0 if modality != "text" else 0.0,
        confidence=0.8,
        execution_time_ms=0,
        metadata={"modality": modality},
    )


# ---------------------------------------------------------------------------
# Public: run all signals in parallel
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    keyword_signal,
    domain_signal,
    complexity_signal,
    language_signal,
    safety_signal,
    pii_signal,
    vision_signal,
    tool_signal,
    modality_signal,
]


async def run_all_signals(messages: list[dict], tools: Optional[list] = None) -> list[SignalResult]:
    """Execute all signals concurrently and return results."""
    tasks = [sig(messages, tools=tools) for sig in ALL_SIGNALS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append(SignalResult(
                name="error", score=0.0, confidence=0.0,
                execution_time_ms=0, metadata={"error": str(r)},
            ))
        else:
            out.append(r)
    return out
