"""Fill missing claude-3-haiku-20240307 ground truth for RouterArena."""
import json
import os
import time
import re
import requests

ANTHROPIC_API_KEY = "os.environ.get("ANTHROPIC_API_KEY", "")"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

CACHE_DIR = os.path.expanduser("~/RouterArena/cached_results")
GPT_FILE = os.path.join(CACHE_DIR, "gpt-4o-mini.jsonl")
CLAUDE_FILE = os.path.join(CACHE_DIR, "claude-3-haiku-20240307.jsonl")

def load_existing():
    existing = set()
    with open(CLAUDE_FILE) as f:
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

def call_claude(prompt_text):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    
    try:
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return text, input_tokens, output_tokens, True
    except Exception as e:
        return str(e), 0, 0, False

def evaluate_answer(generated, ground_truth, metric):
    gen_lower = generated.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    if metric == "mcq_accuracy":
        boxed = re.findall(r'\\boxed\{([A-Z])\}', generated)
        if boxed:
            return 1.0 if boxed[-1].upper() == gt_lower.upper() else 0.0
        for pattern in [r'\b([A-D])\b', r'answer is ([A-D])', r'correct.*?([A-D])']:
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
    existing = load_existing()
    all_queries = load_all_queries()
    
    missing = [q for q in all_queries if q["global_index"] not in existing]
    print(f"Existing claude GT: {len(existing)}")
    print(f"Missing: {len(missing)}")
    
    results = []
    errors = 0
    
    for i, query in enumerate(missing):
        prompt = query["question"]
        gt = query["evaluation_result"]["ground_truth"]
        metric = query["evaluation_result"]["metric"]
        
        answer, in_tokens, out_tokens, success = call_claude(prompt)
        
        if success:
            score = evaluate_answer(answer, gt, metric)
            result = {
                "global_index": query["global_index"],
                "question": prompt,
                "llm_selected": "claude-3-haiku-20240307",
                "generated_answer": answer,
                "token_usage": {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                "success": True,
                "provider": "anthropic",
                "error": None,
                "evaluation_result": {
                    "extracted_answer": answer,
                    "ground_truth": gt,
                    "score": score,
                    "metric": metric,
                    "inference_cost": (in_tokens * 0.25 + out_tokens * 1.25) / 1_000_000,
                }
            }
            results.append(result)
        else:
            errors += 1
            if "rate" in str(answer).lower():
                print(f"  Rate limited at query {i+1}, backing off 5s...")
                time.sleep(5)
        
        if (i + 1) % 100 == 0:
            correct = sum(1 for r in results if r["evaluation_result"]["score"] > 0)
            print(f"  [{i+1}/{len(missing)}] Success: {len(results)}, Errors: {errors}, Accuracy so far: {correct/max(len(results),1):.1%}")
            
            with open(CLAUDE_FILE, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            results = []
        
        time.sleep(0.05)  # 5 req/sec to be safe with rate limits
    
    # Flush remaining results
    if results:
        with open(CLAUDE_FILE, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    
    final_count = len(load_existing())
    print(f"\nDone! Claude GT now has {final_count}/8400 queries")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
