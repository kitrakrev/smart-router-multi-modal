#!/usr/bin/env python3
"""General model server for A100 — serves Llama-3.1-8B + Qwen2.5-7B.

Runs on port 8002 alongside the router (port 8000).
Medical models stay on H200 (port 8001 via tunnel).
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

app = FastAPI(title="General LLM Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS: Dict[str, dict] = {}


class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.95


MODEL_CONFIGS = [
    {
        "name": "llava-1.6-7b",
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "aliases": ["llava-1.6-7b", "llava", "llava-hf/llava-v1.6-mistral-7b-hf"],
        "vision": True,
    },
]

MODEL_ALIASES: Dict[str, str] = {}


def load_model(name: str, model_id: str, is_vision: bool = False):
    print(f"[INFO] Loading {model_id}...")
    start = time.time()

    if is_vision:
        # Vision models need specific model class
        from transformers import LlavaNextForConditionalGeneration
        tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    elapsed = time.time() - start
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"[INFO] {name} loaded in {elapsed:.1f}s, GPU mem: {mem:.1f}GB")
    print(f"[OK] {name} ready!")

    return {"model": model, "tokenizer": tokenizer, "model_id": model_id, "type": "multimodal" if is_vision else "text"}


def generate(model_info: dict, messages: List[ChatMessage], max_tokens: int, temperature: float, top_p: float):
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    text_messages = [{"role": m.role, "content": m.content if isinstance(m.content, str) else " ".join(
        p.get("text", "") for p in m.content if isinstance(p, dict) and p.get("type") == "text"
    ) if isinstance(m.content, list) else str(m.content)} for m in messages]

    try:
        prompt = tokenizer.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in text_messages) + "\nassistant: "

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    elapsed = time.time() - t0

    print(f"[INFO] Generated {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/max(elapsed,0.001):.1f} tok/s)")
    return response_text, input_len, len(generated)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    model_key = MODEL_ALIASES.get(request.model, request.model)
    if not model_key:
        model_key = list(MODELS.keys())[0] if MODELS else ""

    if model_key not in MODELS:
        available = list(MODELS.keys())
        raise HTTPException(404, f"Model '{request.model}' not found. Available: {available}")

    try:
        text, prompt_tokens, completion_tokens = generate(
            MODELS[model_key], request.messages,
            request.max_tokens, request.temperature, request.top_p,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation error: {e}")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or model_key,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": k, "object": "model", "owned_by": "local"} for k in MODELS
    ]}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "gpu": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        },
    }


@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("General LLM Server starting up...")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    for cfg in MODEL_CONFIGS:
        try:
            info = load_model(cfg["name"], cfg["model_id"], is_vision=cfg.get("vision", False))
            MODELS[cfg["name"]] = info
            for alias in cfg["aliases"]:
                MODEL_ALIASES[alias] = cfg["name"]
        except Exception as e:
            print(f"[SKIP] {cfg['name']}: {e}")

    print(f"\nModels loaded: {list(MODELS.keys())}")
    print(f"Total GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
