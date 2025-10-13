#!/usr/bin/env python3
"""
evaluate_specbench.py

Usage example:
python evaluation/evaluate_specbench.py \
  --code-path "inference.py" \
  --specbench "data/SpecBench_question.jsonl" \
  --out "/home/iitb/Kishan_SpecDec/results/specbench_results_gm6_tgt8B_genLen256.jsonl" \
  --summary "/home/iitb/Kishan_SpecDec/specbench_summary.json" \
  --device "cuda:0" \
  --gamma 6 \
  --max-gen-len 256 \
  --max-examples 480 \
  --target-model "meta-llama/Llama-3.1-8B-Instruct" \
  --drafter-model "meta-llama/Llama-3.2-1B-Instruct"

"""


import time
import json
import os
import argparse
import importlib.util
import sys
import math
from typing import List, Optional

import random
import torch
import numpy as np
from tqdm import tqdm


# load module by file
def load_user_module(module_path: str, mod_name: str = "user_specdec"):
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

# Prompt preparation (use tokenizer.apply_chat_template )
def prepare_prompt_from_turns(turns: List[str], tokenizer, chat=True):
    # turns is a list of strings; joining  them for SpecBench dataset.
    try:
        if chat and hasattr(tokenizer, "apply_chat_template"):
            # if there are several turns, join them separated by two newlines
            prompt_text = "\n\n".join(turns)
            # For generality we pass the whole combined text as a single user message.
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
    except Exception:
        # fall back to join below
        pass
    # fallback: simply join the turns with blank lines
    return "\n\n".join(turns)


# Evaluation driver

def evaluate(
    code_path: str,
    specbench_path: str,
    out_jsonl: str,
    summary_json: str,
    device: str = "cuda",
    gamma: int = 4,
    max_gen_len: int = 64,
    processor_name: str = "greedy",
    processor_args = {"temperature": 1.0},
    max_examples: int | None = None,
    seed: int = 42,
    target_model: Optional[str] = None,
    drafter_model: Optional[str] = None,
):
    assert os.path.exists(code_path), f"Code file not found: {code_path}"
    assert os.path.exists(specbench_path), f"SpecBench file not found: {specbench_path}"

    # 1) import user's file as module
    mod = load_user_module(code_path, "specdec_usercode")

    # 2) instantiate the InferenceCLI (this will load models)
    print("Instantiating InferenceCLI (this will load models) ...")
    cli = mod.InferenceCLI(device=device, target_model=target_model, drafter_model=drafter_model)  # this prints info and loads models
    target = cli.target
    drafter = cli.drafter
    tokenizer = cli.tokenizer

    # 3) build processor map (use classes from user module)
    processors = {
        "greedy": mod.GreedyProcessor,
    }
    
    if processor_name not in processors:
        raise ValueError(f"Processor {processor_name} not available. Choose from {list(processors.keys())}")
    processor_class = processors[processor_name]
    processor = processor_class(**processor_args)

    # 4) load SpecBench dataset
    records = []
    with open(specbench_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    n_total = len(records)
    if max_examples is None:
        max_examples = n_total

    # 5) prepare output file
    out_dir = os.path.dirname(out_jsonl) or "."
    os.makedirs(out_dir, exist_ok=True)
    fout = open(out_jsonl, "w", encoding="utf-8")

    # stats accumulators (kept for summary)
    baseline_target_throughputs = []
    baseline_target_num_tokens = []
    baseline_draft_throughputs = []
    baseline_draft_num_tokens = []
    spec_throughputs = []
    spec_num_tokens_list = []
    target_calls_list = []
    acceptance_rates = []

    successes = 0
    failures = 0

    # helper for deterministic generation
    def set_seed_local(s):
        try:
            cli._set_seed(s)
        except Exception:
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

    # iterate
    for i, rec in enumerate(tqdm(records[:max_examples], desc="Examples")):
        try:
            qid = rec.get("question_id", i)
            category = rec.get("category", "unknown")
            turns = rec.get("turns", [])
            prompt_text = prepare_prompt_from_turns(turns, tokenizer, chat=True)

            # Tokenize to ids (list)
            tokenized = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            # ---------- Baseline (autoregressive on target) ----------
            set_seed_local(seed)
            t0 = time.time()
            ar_target_ids = mod.autoregressive_generate(
                tokenized,
                target,
                max_gen_len=max_gen_len,
                logits_processor=processor,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,
            )
            t1 = time.time()
            ar_target_time = t1 - t0
            if ar_target_ids is None:
                ar_target_ids = []
            ar_target_text = tokenizer.decode(ar_target_ids, skip_special_tokens=True)
            ar_target_num_tokens = len(ar_target_ids)
            ar_target_throughput = ar_target_num_tokens / max(ar_target_time, 1e-9)

            # ---------- Baseline (autoregressive on drafter) ----------
            set_seed_local(seed)
            t0 = time.time()
            ar_draft_ids = mod.autoregressive_generate(
                tokenized,
                drafter,
                max_gen_len=max_gen_len,
                logits_processor=processor,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,
            )
            t1 = time.time()
            ar_draft_time = t1 - t0
            if ar_draft_ids is None:
                ar_draft_ids = []
            ar_draft_text = tokenizer.decode(ar_draft_ids, skip_special_tokens=True)
            ar_draft_num_tokens = len(ar_draft_ids)
            ar_draft_throughput = ar_draft_num_tokens / max(ar_draft_time, 1e-9)

            # ---------- Speculative ----------

            set_seed_local(seed)
            t0 = time.time()
            spec_output_ids, accept_rate, target_calls_reported = mod.speculative_generate(
                tokenized,
                drafter,
                target,
                tokenizer=tokenizer,
                gamma=gamma,
                logits_processor=processor,
                max_gen_len=max_gen_len,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,
            )
            t1 = time.time()
            spec_total_time = t1 - t0

            if spec_output_ids is None:
                spec_output_ids = []
            spec_text = tokenizer.decode(spec_output_ids, skip_special_tokens=True)
            spec_num_tokens = len(spec_output_ids)
            spec_throughput = spec_num_tokens / max(spec_total_time, 1e-9)
            measured_target_calls = target_calls_reported


            # accumulate stats
            baseline_target_throughputs.append(ar_target_throughput)
            baseline_target_num_tokens.append(ar_target_num_tokens)
            baseline_draft_throughputs.append(ar_draft_throughput)
            baseline_draft_num_tokens.append(ar_draft_num_tokens)
            spec_throughputs.append(spec_throughput)
            spec_num_tokens_list.append(spec_num_tokens)
            target_calls_list.append(int(measured_target_calls))
            acceptance_rates.append(float(accept_rate) if accept_rate is not None else None)
            

            # write per-prompt result
            item = {
                "question_id": qid,
                "category": category,
                "prompt_text": prompt_text,
                "baseline_target": {
                    "output_text": ar_target_text,
                    "num_tokens": ar_target_num_tokens,
                    "time_sec": ar_target_time,
                    "throughput_toks_per_sec": ar_target_throughput,
                },
                "baseline_draft": {
                    "output_text": ar_draft_text,
                    "num_tokens": ar_draft_num_tokens,
                    "time_sec": ar_draft_time,
                    "throughput_toks_per_sec": ar_draft_throughput,
                },
                "speculative": {
                    "output_text": spec_text,
                    "num_tokens": spec_num_tokens,
                    "time_sec_total": spec_total_time,
                    "throughput_toks_per_sec": spec_throughput,
                    "acceptance_rate": float(accept_rate) if accept_rate is not None else None,
                    "target_calls_measured_wrapper": measured_target_calls,
                    "gamma": gamma,
                },
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
            successes += 1

        except Exception as e:
            failures += 1
            print(f"[ERROR] example index {i} failed: {e}")
            err_obj = {"question_id": rec.get("question_id", i), "error": str(e)}
            fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            fout.flush()
            continue

    fout.close()


    # After processing all examples: compute summary means and write summary_json
    def safe_mean(lst):
        if not lst:
            return None
        try:
            return float(np.mean(lst))
        except Exception:
            # fallback: convert to list of floats ignoring None
            filtered = [float(x) for x in lst if x is not None]
            if not filtered:
                return None
            return float(np.mean(filtered))

    # filter acceptance_rates None values
    acceptance_filtered = [x for x in acceptance_rates if x is not None]

    summary = {
        "n_total": n_total,
        "n_processed": successes + failures,
        "successes": successes,
        "failures": failures,
        "mean_baseline_target_throughput": safe_mean(baseline_target_throughputs),
        "mean_baseline_target_num_tokens": safe_mean(baseline_target_num_tokens),
        "mean_baseline_draft_throughput": safe_mean(baseline_draft_throughputs),
        "mean_baseline_draft_num_tokens": safe_mean(baseline_draft_num_tokens),
        "mean_spec_throughput": safe_mean(spec_throughputs),
        "mean_spec_num_tokens": safe_mean(spec_num_tokens_list),
        "mean_target_calls": safe_mean(target_calls_list),
        "mean_acceptance_rate": safe_mean(acceptance_filtered),
    }

    # Ensure summary directory exists
    summary_dir = os.path.dirname(summary_json) or "."
    os.makedirs(summary_dir, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    # Print the summary to the terminal
    print("\n=== Run summary (means) ===")
    print(json.dumps(summary, indent=2))
    print("===========================\n")




# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", required=True, help="Path to your speculative code file (python), e.g. '/mnt/data/llama_3_2_1b_specdec_v2 (1).py'")
    parser.add_argument("--specbench", required=True, help="Path to SpecBench jsonl")
    parser.add_argument("--out", required=True, help="Output JSONL file for per-prompt results")
    parser.add_argument("--summary", required=True, help="Output JSON summary file for means")
    parser.add_argument("--device", default="cuda", help="Device to pass to InferenceCLI (cuda/cpu)")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma (drafts) for speculative decoding")
    parser.add_argument("--max-gen-len", type=int, default=64, help="Max generation length")
    parser.add_argument("--processor", default="greedy", choices=["greedy"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-model", type=str, default=None, help="HuggingFace repo id or local path to use as TARGET model (overrides specdec defaults)")
    parser.add_argument("--drafter-model", type=str, default=None, help="HuggingFace repo id or local path to use as DRAFTER model (overrides specdec defaults)")

    args = parser.parse_args()

    evaluate(
        code_path=args.code_path,
        specbench_path=args.specbench,
        out_jsonl=args.out,
        summary_json=args.summary,
        device=args.device,
        gamma=args.gamma,
        max_gen_len=args.max_gen_len,
        processor_name=args.processor,
        processor_args={"temperature": 1.0},
        max_examples=args.max_examples,
        seed=args.seed,
        target_model=args.target_model,
        drafter_model=args.drafter_model,
    )
