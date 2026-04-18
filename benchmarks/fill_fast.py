"""Fill missing claude-3-haiku ground truth. Hard cap $4."""
import json, os, time, re, requests, sys

KEY = "REDACTED"
URL = "https://api.anthropic.com/v1/messages"
CACHE = os.path.expanduser("~/RouterArena/cached_results")
CF = os.path.join(CACHE, "claude-3-haiku-20240307.jsonl")
MAX_SPEND = 4.0

existing = set()
with open(CF) as f:
    for l in f:
        existing.add(json.loads(l)["global_index"])

with open(os.path.join(CACHE, "gpt-4o-mini.jsonl")) as f:
    allq = [json.loads(l) for l in f]

missing = [q for q in allq if q["global_index"] not in existing]
print(f"Existing: {len(existing)}, Missing: {len(missing)}", flush=True)

total_cost = 0.0
results = []
errors = 0

for i, q in enumerate(missing):
    if total_cost >= MAX_SPEND:
        print(f"HARD CAP ${MAX_SPEND} reached at query {i}. Stopping.", flush=True)
        break
    try:
        r = requests.post(URL,
            headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-3-haiku-20240307", "max_tokens": 1024, "temperature": 0.1,
                  "messages": [{"role": "user", "content": q["question"]}]},
            timeout=30)
        d = r.json()
        if "content" in d:
            txt = d["content"][0]["text"]
            gt = q["evaluation_result"]["ground_truth"]
            metric = q["evaluation_result"]["metric"]
            score = 0.0
            if metric == "mcq_accuracy":
                m = re.findall(r"\\boxed\{([A-Z])\}", txt) or re.findall(r"\b([A-D])\b", txt)
                if m and m[-1].upper() == gt.upper():
                    score = 1.0
            elif gt.lower() in txt.lower():
                score = 1.0
            usage = d.get("usage", {})
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cost = (inp * 0.25 + out * 1.25) / 1_000_000
            total_cost += cost
            results.append(json.dumps({
                "global_index": q["global_index"],
                "question": q["question"],
                "llm_selected": "claude-3-haiku-20240307",
                "generated_answer": txt,
                "token_usage": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out},
                "success": True, "provider": "anthropic", "error": None,
                "evaluation_result": {"ground_truth": gt, "score": score, "metric": metric, "inference_cost": cost}
            }))
        else:
            errors += 1
            if "rate" in str(d).lower():
                time.sleep(5)
    except Exception as e:
        errors += 1

    if (i + 1) % 50 == 0:
        acc = sum(1 for r in results if json.loads(r)["evaluation_result"]["score"] > 0) / max(len(results), 1)
        print(f"  [{i+1}/{len(missing)}] done={len(results)} err={errors} acc={acc:.0%} cost=${total_cost:.2f}", flush=True)
        with open(CF, "a") as f:
            for r in results:
                f.write(r + "\n")
        results = []
    time.sleep(0.05)

if results:
    with open(CF, "a") as f:
        for r in results:
            f.write(r + "\n")

with open(CF) as f:
    total = sum(1 for _ in f)
print(f"Done! Claude GT: {total}/8400, Errors: {errors}, Total cost: ${total_cost:.2f}", flush=True)
