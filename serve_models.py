#!/usr/bin/env python3
"""
Medical LLM Model Server
Serves medalpaca-7b via OpenAI-compatible API on port 8001.
Designed for NVIDIA H200 MIG 1g.35gb slice (~33GB VRAM).

Models served:
  - medalpaca-7b (primary, ~14GB FP16)
  - MedGemma-4B (if HF auth is configured for gated model access)
"""

import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, LlamaTokenizer

app = FastAPI(title="Medical LLM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model registry
MODELS: Dict[str, Any] = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Pydantic models for OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# --- Model loading ---

def load_llava_med():
    """Load LLaVA-Med-7B — medical vision specialist (pathology + radiology)."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoConfig
    model_id = "microsoft/llava-med-v1.5-mistral-7b"
    print(f"[INFO] Loading {model_id}...")
    start = time.time()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # llava_mistral arch → use LlavaForConditionalGeneration with config override
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.model_type = "llava"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[WARN] LlavaForConditionalGeneration failed: {e}, trying AutoModel")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    elapsed = time.time() - start
    print(f"[INFO] llava-med-7b loaded in {elapsed:.1f}s")

    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"[INFO] GPU memory allocated: {mem_used:.1f}GB, reserved: {mem_reserved:.1f}GB")

    return {
        "model": model,
        "tokenizer": processor,
        "processor": processor,
        "model_id": model_id,
        "type": "multimodal",
    }


def load_medgemma():
    """Load MedGemma-4B (requires HF auth for gated model)."""
    from transformers import AutoProcessor
    model_id = "google/medgemma-4b-it"
    print(f"[INFO] Loading {model_id}...")
    start = time.time()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    elapsed = time.time() - start
    print(f"[INFO] MedGemma loaded in {elapsed:.1f}s")

    return {
        "model": model,
        "tokenizer": processor,
        "processor": processor,
        "model_id": model_id,
        "type": "multimodal",
    }


def build_prompt(messages: List[ChatMessage]) -> str:
    """Build a prompt from chat messages for medalpaca-style models."""
    prompt_parts = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            # Extract text from multimodal content
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            content = " ".join(texts)

        if msg.role == "system":
            prompt_parts.append(f"### System:\n{content}\n")
        elif msg.role == "user":
            prompt_parts.append(f"### Input:\n{content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"### Response:\n{content}\n")

    # Add generation prompt
    prompt_parts.append("### Response:\n")
    return "\n".join(prompt_parts)


def generate_completion(model_info: dict, messages: List[ChatMessage],
                        max_tokens: int, temperature: float, top_p: float):
    """Generate a completion from the model."""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    # Extract text and images from messages
    text_parts = []
    images = []
    for m in messages:
        if isinstance(m.content, str):
            text_parts.append(m.content)
        elif isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            try:
                                img_b64 = url.split(",", 1)[1] if "," in url else url
                                img_bytes = base64.b64decode(img_b64)
                                from PIL import Image as PILImage
                                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                                images.append(img)
                            except Exception:
                                pass

    # Build messages for chat template
    is_multimodal_model = model_info.get("type") == "multimodal"
    text_messages = []
    for m in messages:
        if isinstance(m.content, str):
            text_messages.append({"role": m.role, "content": m.content})
        elif isinstance(m.content, list):
            if is_multimodal_model and images:
                # For multimodal models (MedGemma): use {"type":"image"} format
                content_parts = []
                for block in m.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_parts.append({"type": "text", "text": block.get("text", "")})
                        elif block.get("type") == "image_url":
                            content_parts.insert(0, {"type": "image"})
                text_messages.append({"role": m.role, "content": content_parts})
            else:
                # Text-only: just join text parts
                parts = [block.get("text", "") for block in m.content
                         if isinstance(block, dict) and block.get("type") == "text"]
                text_messages.append({"role": m.role, "content": " ".join(parts)})

    try:
        template_fn = getattr(tokenizer, "apply_chat_template", None)
        if template_fn is None and hasattr(tokenizer, "tokenizer"):
            template_fn = getattr(tokenizer.tokenizer, "apply_chat_template", None)
        if template_fn:
            prompt = template_fn(
                text_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = build_prompt(messages)
    except Exception:
        prompt = build_prompt(messages)

    start = time.time()

    # Handle multimodal (MedGemma with images) vs text-only
    is_processor = hasattr(tokenizer, "image_processor")
    if is_processor and images:
        # Multimodal: pass image + text through processor
        print(f"[DEBUG] Multimodal prompt has <image>: {'<image>' in prompt}, len={len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:200]}")
        inputs = tokenizer(text=prompt, images=images[0], return_tensors="pt", truncation=True, max_length=2048)
    elif is_processor:
        inputs = tokenizer(text=prompt, return_tensors="pt", truncation=True, max_length=2048)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None and hasattr(tokenizer, "tokenizer"):
            pad_id = getattr(tokenizer.tokenizer, "eos_token_id", 0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=pad_id or 0,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    decode_fn = tokenizer.decode if hasattr(tokenizer, "decode") else tokenizer.tokenizer.decode
    response_text = decode_fn(generated, skip_special_tokens=True).strip()

    elapsed = time.time() - start
    prompt_tokens = input_len
    completion_tokens = len(generated)

    print(f"[INFO] Generated {completion_tokens} tokens in {elapsed:.2f}s "
          f"({completion_tokens / max(elapsed, 0.001):.1f} tok/s)")

    return response_text, prompt_tokens, completion_tokens


# --- API endpoints ---

@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = []
    for name, info in MODELS.items():
        models.append({
            "id": name,
            "object": "model",
            "owned_by": "local",
            "model_id": info["model_id"],
            "type": info["type"],
        })
    return {"object": "list", "data": models}


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        }
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "gpu": gpu_info,
    }


# Model name aliases -> internal key
MODEL_ALIASES = {
    "llava-med-7b": "llava-med",
    "llava-med": "llava-med",
    "microsoft/llava-med-v1.5-mistral-7b": "llava-med",
    "medgemma-4b": "medgemma",
    "medgemma": "medgemma",
    "google/medgemma-4b-it": "medgemma",
    "google/medgemma-1.5-4b-it": "medgemma",
}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    model_key = MODEL_ALIASES.get(request.model, request.model)

    if model_key not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {list(MODELS.keys())}",
        )

    model_info = MODELS[model_key]

    try:
        response_text, prompt_tokens, completion_tokens = generate_completion(
            model_info, request.messages,
            request.max_tokens, request.temperature, request.top_p,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": response_text},
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    print("=" * 60)
    print("Medical LLM Server starting up...")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"VRAM: {vram:.1f} GB")
    print("=" * 60)

    # Load LLaVA-Med-7B (pathology + radiology specialist, ~14GB FP16)
    try:
        MODELS["llava-med"] = load_llava_med()
        print("[OK] llava-med-7b ready!")
    except Exception as e:
        print(f"[ERROR] Failed to load llava-med: {e}")
        import traceback
        traceback.print_exc()

    # Try loading MedGemma (requires HF gated model access)
    try:
        MODELS["medgemma"] = load_medgemma()
        print("[OK] MedGemma ready!")
    except Exception as e:
        print(f"[SKIP] MedGemma not available (gated model, need HF auth): {type(e).__name__}")

    print("=" * 60)
    print(f"Models loaded: {list(MODELS.keys())}")
    print("Server ready on http://0.0.0.0:8001")
    print("=" * 60)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
