"""Fill missing gemini-2.0-flash-001 ground truth for RouterArena.
Resilient to rate limits - will retry with exponential backoff.
Resumable - reloads existing entries each time a batch is written."""
import json
import os
import sys
import time
import re
import requests
from datetime import datetime

GEMINI_API_KEY = "os.environ.get("GOOGLE_API_KEY", "")"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={GEMINI_API_KEY}"

CACHE_DIR = os.path.expanduser("~/RouterArena/cached_results")
GPT_FILE = os.path.join(CACHE_DIR, "gpt-4o-mini.jsonl")
GEMINI_FILE = os.path.join(CACHE_DIR, "gemini-2.0-flash-001.jsonl")

def load_existing_gemini():
    existing = set()
    with open(GEMINI_FILE) as f:
        for line in f:
            row = json.loads(line)
            existing.add(row["global_index"])
    return existing

def load_all_queries():
    queries = []
    with open(GPT_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def call_gemini(prompt_text, max_retries=8):
    """Call Gemini API with aggressive backoff for rate limits."""
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.1,
        }
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=60)
            
            if resp.status_code == 429:
                # Exponential backoff: 15, 30, 60, 120, 240, 480, 960, 1920
                delay = min(15 * (2 ** attempt), 1920)
                now = datetime.now().strftime("%H:%M:%S")
                print(f"    [{now}] Rate limited, waiting {delay}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(delay)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            candidates = data.get("candidates", [])
            if not candidates:
                return "NO_CANDIDATES", 0, 0, False
            
            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason == "SAFETY":
                return "SAFETY_BLOCKED", 0, 0, False
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts or "text" not in parts[0]:
                return "NO_TEXT", 0, 0, False
            
            text = parts[0]["text"]
            usage = data.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            
            return text, input_tokens, output_tokens, True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return str(e), 0, 0, False
    
    return "MAX_RETRIES_EXCEEDED", 0, 0, False

def evaluate_answer(generated, ground_truth, metric):
    gen_lower = generated.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    if metric == "mcq_accuracy":
        boxed = re.findall(r'\\boxed\{([A-Z])\}', generated)
        if boxed:
            return 1.0 if boxed[-1].upper() == gt_lower.upper() else 0.0
        for pattern in [r'answer is ([A-D])', r'correct.*?([A-D])', r'\b([A-D])\b']:
            match = re.findall(pattern, generated, re.IGNORECASE)
            if match:
                return 1.0 if match[-1].upper() == gt_lower.upper() else 0.0
        return 0.0
    elif metric == "math_metric":
        numbers = re.findall(r'\\boxed\{([^}]+)\}', generated)
        if numbers:
            return 1.0 if numbers[-1].strip() == gt_lower.strip() else 0.0
        return 0.0
    elif metric == "exact_match":
        return 1.0 if gt_lower in gen_lower else 0.0
    else:
        return 1.0 if gt_lower in gen_lower else 0.0

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...", flush=True)
    existing = load_existing_gemini()
    all_queries = load_all_queries()
    
    missing = [q for q in all_queries if q["global_index"] not in existing]
    print(f"Existing gemini GT: {len(existing)}", flush=True)
    print(f"Missing: {len(missing)}", flush=True)
    
    if not missing:
        print("All queries already have gemini GT!", flush=True)
        return
    
    results = []
    errors = 0
    skipped = 0
    total_processed = 0
    
    for i, query in enumerate(missing):
        prompt = query["question"]
        gt = query["evaluation_result"]["ground_truth"]
        metric = query["evaluation_result"]["metric"]
        
        answer, in_tokens, out_tokens, success = call_gemini(prompt)
        
        if success:
            score = evaluate_answer(answer, gt, metric)
            
            result = {
                "global_index": query["global_index"],
                "question": prompt,
                "llm_selected": "gemini-2.0-flash-001",
                "generated_answer": answer,
                "token_usage": {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                "success": True,
                "provider": "google",
                "error": None,
                "evaluation_result": {
                    "extracted_answer": answer,
                    "ground_truth": gt,
                    "score": score,
                    "metric": metric,
                    "inference_cost": (in_tokens * 0.1 + out_tokens * 0.4) / 1_000_000,
                }
            }
            results.append(result)
        else:
            if answer == "MAX_RETRIES_EXCEEDED":
                # Quota truly exhausted for the day; skip rest
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Quota exhausted after all retries. Stopping.", flush=True)
                errors += 1
                break
            errors += 1
            print(f"  ERROR on {query['global_index']}: {answer[:80]}", flush=True)
        
        total_processed += 1
        
        if total_processed % 50 == 0 or total_processed == len(missing):
            correct = sum(1 for r in results if r["evaluation_result"]["score"] > 0)
            total_results = len(results)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [{total_processed}/{len(missing)}] "
                  f"Success: {total_results}, Errors: {errors}, "
                  f"Accuracy: {correct/max(total_results,1):.1%}", flush=True)
            
            if results:
                with open(GEMINI_FILE, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                results = []
        
        # Rate limit: be gentle
        time.sleep(0.5)
    
    # Write remaining
    if results:
        with open(GEMINI_FILE, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    
    final_count = len(load_existing_gemini())
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done! Gemini GT now has {final_count}/8400 queries", flush=True)
    print(f"Errors: {errors}", flush=True)

if __name__ == "__main__":
    main()
