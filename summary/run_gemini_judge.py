#!/usr/bin/env python3
"""
run_gemini_judge.py

This is the same runner as before but with a robust call_gemini() that:
- probes multiple SDK entrypoints & argument shapes (covers different google-genai releases)
- if SDK attempts fail, tries several REST endpoint variants (v1beta1 / v1beta) and several request body shapes
- logs detailed attempt errors for debugging
- returns the raw generated text (expected to be JSON) or raises an informative error

Usable
    export GEMINI_API_KEY=" "
    python run_gemini_judge.py


"""

import os
import json
import time
import math
import random
import inspect
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import numpy as np
import requests

# Try to import google-genai SDK (may or may not be installed)
try:
    from google import genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

#  CONFIG 
INPUT_PATH = "/home/iitb/Kishan_SpecDec/results_specbench/specbench_results_gm8_tgt8B_genLen256_cleaned.jsonl"   # input path (JSONL or JSON array)
OUTPUT_JUDGED_PATH = "/home/iitb/Kishan_SpecDec/results_specbench/specbench_summary_gm8_tgt8B_genLen256_cleaned.jsonl"
AGGREGATE_SUMMARY_PATH = "/home/iitb/Kishan_SpecDec/summary_specbench/specbench_aggregateSummary_gm8_tgt8B_genLen256_cleaned.jsonl"
GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_TOKENS_GLOBAL = 256

QUOTA_RPM = 15
QUOTA_TPM = 250_000
QUOTA_RPD = 1000
SLEEP_BETWEEN_CALLS = None
MAX_RETRIES = 6
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42
MAX_OUTPUT_TOKENS = 700
TEMPERATURE = 0.0
# --------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set environment variable GEMINI_API_KEY before running (GEMINI_API_KEY).")

# initialize SDK client if available
client = None
if _HAS_GENAI:
    try:
        client = genai.Client()
        print("[INFO] google-genai SDK: initialized client")
    except Exception as e:
        print("[WARN] google-genai SDK import succeeded but client init failed:", e)
        client = None
else:
    print("[INFO] google-genai SDK not available; will try REST fallback.")

def compute_safe_sleep(tokens_per_call_estimate: float = 800.0):
    sleep_by_rpm = 60.0 / max(1, QUOTA_RPM)
    sleep_by_tpm = (tokens_per_call_estimate / max(1, QUOTA_TPM)) * 60.0
    return max(sleep_by_rpm, sleep_by_tpm)

if SLEEP_BETWEEN_CALLS is None:
    SLEEP_BETWEEN_CALLS = compute_safe_sleep(2700.0)

print(f"[INFO] Using Gemini model: {GEMINI_MODEL}")
print(f"[INFO] Throttle sleep per call: {SLEEP_BETWEEN_CALLS:.2f}s (RPM={QUOTA_RPM}, TPM={QUOTA_TPM})")
print(f"[INFO] Daily request limit (RPD) = {QUOTA_RPD}")

#  (load, normalize, build prompt) 
def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("Input JSON must be an array or JSONL.")
    except json.JSONDecodeError:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                items.append(json.loads(line))
        return items

def normalize_category(cat: Optional[str]) -> str:
    if cat is None:
        return "unknown"
    s = cat.strip().lower()
    mapping = {
        "multi_turn_conversation": "multi_turn_conversation",
        "translation": "translation", 
        "summarization": "summarization",
        "qa": "qa",
        "math_reasoning": "math_reasoning",
        "rag": "rag", 
    }
    return mapping.get(s, s)

def build_judge_prompt(item: Dict[str, Any]) -> str:
    prompt_id = item.get("question_id")
    category = normalize_category(item.get("category", "unknown"))
    draft_text = item.get("draftonlyoutput", "")
    target_text = item.get("targetonlyoutput", "")
    spec_text = item.get("specbenchoutput", "")
    draft_toks = item.get("draft_token_count", None)
    target_toks = item.get("target_token_count", None)
    spec_toks = item.get("spec_token_count", None)
    tokenizer_name = "meta-llama/Llama-3.1-8B" 
    draft_trunc = None if draft_toks is None else (draft_toks >= MAX_TOKENS_GLOBAL)
    target_trunc = None if target_toks is None else (target_toks >= MAX_TOKENS_GLOBAL)
    spec_trunc = None if spec_toks is None else (spec_toks >= MAX_TOKENS_GLOBAL)

    prompt = f"""
You are an objective evaluator. Return ONLY a single JSON object (no extra text) that matches the schema:
{{ "prompt_id": "<string|int>", "category": "<multi_turn_conversation|translation|summarization|qa|math_reasoning|rag>",
  "max_tokens": <int>, "evaluations": {{ "draft": <EvaluationObject>, "target": <EvaluationObject>, "spec": <EvaluationObject> }} }}

Each EvaluationObject must include generation.response_tokens, generation.truncated_flag, tokenizers, numeric scores (1-5), overall, adjusted_overall or null, optimality_under_limit, error_tags, limited_by_max_tokens (bool|null), confidence (0..1), and a brief comment.

Apply this aggregation:
 1) Per-evaluation overall:
    overall_score_raw = 0.28*correctness
                      + 0.18*factuality
                      + 0.14*instruction_following
                      + 0.12*completeness
                      + 0.10*clarity_coherence
                      + 0.08*conciseness
                      + 0.10*optimality_under_limit

    overall = round(overall_score_raw)  (then clamp to the integer range 1..5)

 2) Token-limit adjustment:
    If limited_by_max_tokens == true, compute
      completeness_adj = min(5, completeness + 1)
    then recompute the weighted sum using completeness_adj to obtain
      adjusted_overall_raw
    and set
      adjusted_overall = round(adjusted_overall_raw) (clamped to 1..5).
    (Keep both `overall` and `adjusted_overall` fields.)

 3) Mean overall across the three evaluations (draft, target, spec):
    - For each evaluation (draft/target/spec) pick the effective overall:
        effective_overall = adjusted_overall if adjusted_overall is not null
                          else overall
    - Let N be the number of evaluations present with an effective_overall.
      If N == 0 then mean_overall = null.
    - Otherwise compute the arithmetic mean:
        mean_overall = (sum of effective_overall across present evals) / N
      Report mean_overall as a decimal rounded to two decimal places.
    - Also include mean_overall_rounded = round(mean_overall) clamped to 1..5

 4) Critical-failure override (optional best practice):
    If correctness <= 1 OR factuality <= 1 then set overall (and adjusted_overall if computed) = 1 regardless of the weighted sum.

Return the fields `overall`, `adjusted_overall` (if applicable), `mean_overall` (float, 2 d.p.), and `mean_overall_rounded` (int, 1..5).


Evaluate each response GIVEN max_tokens = {MAX_TOKENS_GLOBAL}. If response_tokens >= max_tokens or truncation obvious, set truncated_flag=true, add 'truncated' to error_tags, and mark limited_by_max_tokens accordingly.

Return EXACTLY the JSON object and NOTHING else.

PROMPT_ID: {prompt_id}
CATEGORY: {category}
USER_PROMPT: {item.get('prompt_text','')}

--- DRAFT ---
TEXT: {draft_text}
GENERATION:
 response_tokens: {json.dumps(draft_toks)}
 truncated_flag: {json.dumps(draft_trunc)}
 tokenizer: {json.dumps(tokenizer_name)}

--- TARGET ---
TEXT: {target_text}
GENERATION:
 response_tokens: {json.dumps(target_toks)}
 truncated_flag: {json.dumps(target_trunc)}
 tokenizer: {json.dumps(tokenizer_name)}

--- SPEC ---
TEXT: {spec_text}
GENERATION:
 response_tokens: {json.dumps(spec_toks)}
 truncated_flag: {json.dumps(spec_trunc)}
 tokenizer: {json.dumps(tokenizer_name)}

GLOBAL: max_tokens: {MAX_TOKENS_GLOBAL}
"""
    return prompt

#  call_gemini 
def _extract_text_from_resp(resp) -> Optional[str]:
    """Extract text from various possible SDK/REST response shapes."""
    if resp is None:
        return None
    # str
    if isinstance(resp, str):
        return resp
    # object with text attr
    if hasattr(resp, "text"):
        return resp.text
    # dict-like responses
    if isinstance(resp, dict):
        if "candidates" in resp and resp["candidates"]:
            cand = resp["candidates"][0]
            if isinstance(cand, dict):
                if "content" in cand and isinstance(cand["content"], list) and cand["content"]:
                    part = cand["content"][0]
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]
                if "output" in cand:
                    return cand["output"]
            return json.dumps(cand)
        if "output" in resp:
            return resp["output"]
        if "text" in resp:
            return resp["text"]
        # fallback to stringify
        return json.dumps(resp)
    # try object __dict__
    try:
        d = getattr(resp, "__dict__", None)
        if isinstance(d, dict):
            # try common keys
            for key in ("text","output","candidates"):
                if key in d:
                    return _extract_text_from_resp(d[key])
    except Exception:
        pass
    # lastly, convert to string
    return str(resp)

def call_gemini(prompt: str, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """
    Robust Gemini caller:
      - tries multiple SDK entrypoints & argument shapes
      - if SDK fails, tries several REST endpoints & request body shapes
    Returns the generated text (string).
    Raises RuntimeError with diagnostic info if all paths fail.
    """
    sdk_errors = []
    # --- SDK attempts: try a few likely callables and argument shapes ---
    if client is not None:
        # gather potential callables on client and client.models
        candidates = []
        if hasattr(client, "models"):
            candidates.append(("client.models.generate_content", getattr(client.models, "generate_content", None)))
            candidates.append(("client.models.generate", getattr(client.models, "generate", None)))
            candidates.append(("client.models.create", getattr(client.models, "create", None)))
        # top-level variants
        candidates.append(("client.generate_content", getattr(client, "generate_content", None)))
        candidates.append(("client.generate_text", getattr(client, "generate_text", None)))
        candidates.append(("client.generate", getattr(client, "generate", None)))
        candidates.append(("client.create", getattr(client, "create", None)))
        # dynamic names
        for name in ("generate_content", "generate", "generate_text", "create", "complete"):
            if hasattr(client, name):
                candidates.append((f"client.{name}", getattr(client, name)))

        # prepare a set of candidate kwargs shapes to try per callable
        candidate_payloads = [
            # new genai: model + contents list
            {"model": GEMINI_MODEL, "contents": [{"parts":[{"text": prompt}]}], "temperature": TEMPERATURE, "max_output_tokens": max_output_tokens},
            {"model": GEMINI_MODEL, "contents": [{"text": prompt}], "temperature": TEMPERATURE, "max_output_tokens": max_output_tokens},
            # older shapes
            {"model": GEMINI_MODEL, "prompt": prompt, "temperature": TEMPERATURE, "max_output_tokens": max_output_tokens},
            {"model": GEMINI_MODEL, "input": prompt, "temperature": TEMPERATURE, "maxOutputTokens": max_output_tokens},
            {"model": GEMINI_MODEL, "messages": [{"role": "user", "content":[{"type":"text","text": prompt}]}], "temperature": TEMPERATURE},
            {"model": GEMINI_MODEL, "text": prompt, "temperature": TEMPERATURE},
        ]

        for name, func in candidates:
            if func is None:
                continue
            sig = None
            try:
                sig = inspect.signature(func)
                params = sig.parameters.keys()
            except Exception:
                params = None
            # try each payload, filtering to accepted kwargs
            for payload in candidate_payloads:
                # filter payload to the callable's parameters if we can
                try_kwargs = {}
                if params:
                    for k, v in payload.items():
                        if k in params:
                            try_kwargs[k] = v
                    # if try_kwargs empty, nonetheless attempt calling with the first positional form
                    if not try_kwargs:
                        # if callable accepts single positional arg, try that
                        try:
                            resp = func(prompt)
                            text = _extract_text_from_resp(resp)
                            if text:
                                print(f"[INFO] SDK: used {name} with positional prompt")
                                return text
                        except Exception as expos:
                            sdk_errors.append((name + " positional", str(expos)))
                            continue
                else:
                    try_kwargs = payload
                # attempt call
                try:
                    resp = func(**try_kwargs)
                    text = _extract_text_from_resp(resp)
                    if text:
                        print(f"[INFO] SDK: used {name} with kwargs {list(try_kwargs.keys())}")
                        return text
                except Exception as ex:
                    sdk_errors.append((name + " with_keys=" + ",".join(try_kwargs.keys()), str(ex)))
                    continue

    # If SDK failed or not available -> try REST endpoints & body shapes
    rest_errors = []
    rest_endpoints =  [
        f"https://generativelanguage.googleapis.com/v1beta1/models/{GEMINI_MODEL}:generateText",
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateText",
        f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateText",
        f"https://generativelanguage.googleapis.com/v1beta1/models/{GEMINI_MODEL}:generate",
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generate",
    ]
    rest_bodies = [
        {"prompt": {"text": prompt}, "maxOutputTokens": max_output_tokens, "temperature": TEMPERATURE},
        {"input": prompt, "maxOutputTokens": max_output_tokens, "temperature": TEMPERATURE},
        {"model": GEMINI_MODEL, "contents": [{"type": "text", "text": prompt}], "maxOutputTokens": max_output_tokens, "temperature": TEMPERATURE},
        {"instances": [{"content": prompt}], "maxOutputTokens": max_output_tokens},
    ]
    # Use a session (connection reuse) and set api-key header; optionally use Authorization: Bearer when available.
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY})

    for url in rest_endpoints:
        for body in rest_bodies:
            try:
                r = session.post(url, json=body, timeout=120)
            except Exception as e:
                rest_errors.append((url, str(body), f"exception:{e}"))
                time.sleep(1)
                continue
            if r.status_code == 200:
                try:
                    j = r.json()
                    text = _extract_text_from_resp(j)
                    if text:
                        print(f"[INFO] REST: used endpoint {url} with body keys {list(body.keys())}")
                        return text
                    else:
                        rest_errors.append((url, str(body), f"200_ok_but_no_text: {j}"))
                except Exception as ejson:
                    rest_errors.append((url, str(body), f"200_ok_but_json_err:{ejson}"))
            # 400: invalid payload shape -> log and skip this body (server won't accept this body)
            elif r.status_code == 400:
                rest_errors.append((url, str(body), f"HTTP_400:{r.text}"))
                # do not treat as transient; try next body/endpoint
                continue

            # transient throttling / server error -> respect Retry-After or backoff once, then continue
            elif r.status_code in (429, 502, 503, 500):
                rest_errors.append((url, str(body), f"HTTP_{r.status_code}:{r.text}"))
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                        time.sleep(wait + random.random())
                    except Exception:
                        time.sleep(1 + random.random())
                else:
                    # lightweight fixed/backoff sleep for demo; consider exponential backoff helper
                    time.sleep(min(60, 1 * (2 ** 1)))
                # try next body/endpoint after backoff
                continue

            else:
                # other client errors (401,403,404, etc.) -> record and skip
                rest_errors.append((url, str(body), f"HTTP_{r.status_code}:{r.text}"))
                continue

            # short pause between different bodies to avoid bursts
            time.sleep(0.3)


    # If we reach here, everything failed: raise with diagnostics
    msg = {
        "sdk_errors_sample": sdk_errors[:8],
        "rest_errors_sample": rest_errors[:8],
        "note": "Check model id (GEMINI_MODEL), API key, and whether your project can access this model. If REST returns 404, the model id may be incorrect for your account."
    }
    raise RuntimeError("All Gemini SDK/REST attempts failed. Diagnostics: " + json.dumps(msg, indent=2))

#  minimal validator & aggregation (unchanged) 
def validate_judge_json(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "prompt_id" not in obj or "evaluations" not in obj:
        return False
    if not all(k in obj["evaluations"] for k in ("draft", "target", "spec")):
        return False
    return True

def mean_and_ci(arr: List[float], n_boot=BOOTSTRAP_SAMPLES, alpha=0.05):
    if len(arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    arr = np.array(arr, dtype=float)
    mean = float(np.mean(arr))
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boots.append(np.mean(sample))
    low = float(np.percentile(boots, 100 * (alpha / 2)))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": mean, "ci_low": low, "ci_high": high}



def _safe_num(v):
    """Return float if possible, else None."""
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # handle numeric strings
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def compute_model_summary_for_list(judged_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    models = ["draft", "target", "spec"]
    score_axes = ["correctness", "instruction_following", "completeness", "clarity_coherence", "conciseness", "factuality", "robustness", "efficiency"]
    summary = {}
    # Collect the judge-returned item-level mean_overall once per item
    item_mean_overalls = []
    # Precompute per-item whether it has mean_overall
    for j in judged_list:
        top_mean = _safe_num(j.get("mean_overall"))
        if top_mean is not None:
            item_mean_overalls.append(top_mean)

    for m in models:
        vals_overall, vals_adj, vals_opt = [], [], []
        axis_vals = {ax: [] for ax in score_axes}
        truncated_count = 0
        total = 0
        for j in judged_list:
            ev = j.get("evaluations", {}).get(m)
            if ev is None:
                continue
            total += 1
            gen = ev.get("generation", {}) or {}
            resp_toks = gen.get("response_tokens")
            if resp_toks is not None:
                try:
                    if int(resp_toks) >= MAX_TOKENS_GLOBAL:
                        truncated_count += 1
                except Exception:
                    pass
            # append numeric overall / adjusted_overall safely
            overall_val = _safe_num(ev.get("overall"))
            adj_val = _safe_num(ev.get("adjusted_overall"))
            opt_val = _safe_num(ev.get("optimality_under_limit"))
            if overall_val is not None:
                vals_overall.append(overall_val)
            if adj_val is not None:
                vals_adj.append(adj_val)
            if opt_val is not None:
                vals_opt.append(opt_val)
            for ax in score_axes:
                v = _safe_num(ev.get("scores", {}).get(ax))
                if v is not None:
                    axis_vals[ax].append(v)
        model_summary = {"n": total, "truncated_fraction": (truncated_count / total) if total > 0 else None}
        model_summary["overall"] = mean_and_ci(vals_overall)
        model_summary["adjusted_overall"] = mean_and_ci(vals_adj)
        model_summary["optimality_under_limit"] = mean_and_ci(vals_opt)
        model_summary["per_axis"] = {ax: mean_and_ci(vals) for ax, vals in axis_vals.items()}
        summary[m] = model_summary

    # Add aggregation for the item-level mean_overall (the judge-returned mean across draft/target/spec)
    summary["_item_mean_overall"] = mean_and_ci(item_mean_overalls)
    return summary



def aggregate_results(all_judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_summary = compute_model_summary_for_list(all_judged)
    category_groups: Dict[str, List[Dict[str, Any]]] = {}
    for j in all_judged:
        cat = normalize_category(j.get("category", None))
        category_groups.setdefault(cat, []).append(j)
    by_category = {}
    for cat, items in category_groups.items():
        by_category[cat] = compute_model_summary_for_list(items)
    return {"global": global_summary, "by_category": by_category}

#  main 
def main():
    data = load_input(INPUT_PATH)
    print(f"[INFO] Loaded {len(data)} items")
    judged = []
    processed_ids = set()
    if os.path.exists(OUTPUT_JUDGED_PATH):
        with open(OUTPUT_JUDGED_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec.get("prompt_id") or rec.get("question_id") or rec.get("id")
                    if pid is not None:
                        processed_ids.add(str(pid))
                    judged.append(rec)
                except Exception:
                    continue
        print(f"[INFO] Resuming: loaded {len(processed_ids)} already-processed prompt ids from {OUTPUT_JUDGED_PATH}")

    daily_calls = 0
    start_day = time.strftime("%Y-%m-%d")

    for item in tqdm(data):
        pid = item.get("question_id") or item.get("prompt_id") or item.get("id")
        if pid is None:
            continue
        if str(pid) in processed_ids:
            continue

        category = normalize_category(item.get("category", None))
        today = time.strftime("%Y-%m-%d")
        if today != start_day:
            daily_calls = 0
            start_day = today
        if daily_calls >= QUOTA_RPD:
            print(f"[WARN] Reached daily request limit (RPD={QUOTA_RPD}). Stopping run. Resume later or adjust quota.")
            break

        prompt_text = build_judge_prompt(item)
        try:
            raw = call_gemini(prompt_text, max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            print(f"[ERROR] Gemini call failed for prompt {pid}: {e}")
            err = {"prompt_id": pid, "category": category, "error": str(e)}
            with open(OUTPUT_JUDGED_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(err, ensure_ascii=False) + "\n")
            judged.append(err)
            processed_ids.add(str(pid)) 
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        
        parsed = None
        # Robust JSON extraction: decode the first JSON object in `raw`
        try:
            decoder = json.JSONDecoder()
            raw_str = raw.strip()
            idx = 0
            # find the first { and attempt to decode from there
            first_brace = raw_str.find("{")
            if first_brace == -1:
                raise ValueError("no_json_object_found")
            try:
                obj, end = decoder.raw_decode(raw_str[first_brace:])
                parsed = obj
            except Exception as e_raw:
                # fallback: try to decode from start (in case raw begins with JSON)
                try:
                    obj, end = decoder.raw_decode(raw_str)
                    parsed = obj
                except Exception as e2:
                    # log raw and raise to handled block below
                    raise ValueError(f"json_decode_failed: {e_raw} / {e2}")
        except Exception as e:
            # write parse error with pid and timestamp
            with open("parse_errors.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PROMPT_ID: {pid}\nERROR: {e}\nRAW:\n{raw}\n\n")
            parsed = {"prompt_id": pid, "category": category, "error": "parse_failed", "raw": raw}

        if "category" not in parsed:
            parsed["category"] = category

        # --- Fallback: if model didn't provide top-level mean_overall, compute it from evaluations ---
        def compute_item_mean_from_evals(judged_obj: Dict[str, Any]) -> Optional[float]:
            evs = judged_obj.get("evaluations") or {}
            totals = []
            for key in ("draft", "target", "spec"):
                ev = evs.get(key)
                if not ev:
                    continue
                # prefer adjusted_overall if present, else overall
                eff = ev.get("adjusted_overall", None)
                if eff is None:
                    eff = ev.get("overall", None)
                effn = _safe_num(eff)
                if effn is not None:
                    totals.append(effn)
            if not totals:
                return None
            return round(float(sum(totals)) / len(totals), 2)

        # if model didn't return mean_overall, compute and set mean_overall and rounded field
        if parsed.get("mean_overall") is None:
            computed = compute_item_mean_from_evals(parsed)
            if computed is not None:
                parsed["mean_overall"] = computed
                # also set mean_overall_rounded consistent with prompt
                try:
                    parsed["mean_overall_rounded"] = int(max(1, min(5, round(parsed["mean_overall"]))))
                except Exception:
                    parsed["mean_overall_rounded"] = None



        with open(OUTPUT_JUDGED_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        judged.append(parsed)
        processed_ids.add(str(pid))

        daily_calls += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

    judged_for_agg = [j for j in judged if isinstance(j, dict) and "evaluations" in j]
    summary = aggregate_results(judged_for_agg)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "n_items_total": len(data), "n_judged": len(judged_for_agg), "summary": summary}
    with open(AGGREGATE_SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[INFO] Done. Wrote judged items and aggregate summary.")

if __name__ == "__main__":
    main()
