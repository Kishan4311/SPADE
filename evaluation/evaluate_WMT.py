
#!/usr/bin/env python3
"""
evaluate_WMT.py
cnnDailyMailApproach

Usage example:
python /home/iitb/Kishan_SpecDec/Archived/evaluate_WMT.py \
  --device "cuda:1" \
  --dataset "/home/iitb/Kishan_SpecDec/Data/wmt.json" \
  --specdec_path "/home/iitb/Kishan_SpecDec/Archived/inference.py" \
  --k 100 \
  --gen_len 128 \
  --gamma 4 \
  --target_model "meta-llama/Llama-3.2-3B-Instruct" \
  --drafter_model "meta-llama/Llama-3.2-1B-Instruct" \
  --output "/home/iitb/Kishan_SpecDec/resultsWMT/wmt_result_gm4_tgt3B_genLen128.json"

  python /home/iitb/Kishan_SpecDec/Archived/evaluate_WMT.py \
  --device "cuda:1" \
  --dataset "/home/iitb/Kishan_SpecDec/Data/wmt.json" \
  --specdec_path "/home/iitb/Kishan_SpecDec/Archived/inference.py" \
  --k 2 \
  --gen_len 128 \
  --gamma 4 \
  --target_model "Qwen/Qwen3-8B" \
  --drafter_model "Qwen/Qwen3-0.6B" \
  --output "/home/iitb/Kishan_SpecDec/resultsWMT/testQWEN.json"

"""

import argparse
import importlib.util
import json
import os
import time
from typing import Any, Dict, List
from tqdm import tqdm
import torch
import random


def import_module_from_path(path: str, module_name: str = "specdec_user_module"):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Specdec file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


class TimedModelWrapper:
    def __init__(self, model):
        self._model = model
        self.calls = 0
        self.elapsed = 0.0

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        out = self._model(*args, **kwargs)
        t1 = time.time()
        self.calls += 1
        self.elapsed += (t1 - t0)
        return out

    def __getattr__(self, name):
        return getattr(self._model, name)


def prepare_tokens_from_cli(cli, prompt_text: str) -> List[int]:
    """
    Mirror the CLI's approach for preparing tokens: use chat template if cli.chat True.

    """
    try:
        if getattr(cli, "chat", False):
            
            prompt_wrap = cli.tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
        else:
            prompt_wrap = prompt_text
    except Exception:
        
        prompt_wrap = prompt_text

    tokenized = cli.tokenizer(prompt_wrap, return_tensors="pt").input_ids[0].tolist()
    return tokenized

def evaluate(dataset_path: str,
             specdec_path: str,
             output_path: str,
             k: int = 10,
             device: str = "cuda",
             gen_len: int = 64,
             gamma: int = 4,
             seed: int = 42,
             target_model: str | None = None,
             drafter_model: str | None = None):
    

    print(f"Importing specdec module from: {specdec_path}")
    mod = import_module_from_path(specdec_path)

    # expect these names
    for name in ("InferenceCLI", "speculative_generate", "autoregressive_generate"):
        if not hasattr(mod, name):
            raise AttributeError(f"Imported module does not have required symbol: {name}")

    InferenceCLI = getattr(mod, "InferenceCLI")
    speculative_generate = getattr(mod, "speculative_generate")
    autoregressive_generate = getattr(mod, "autoregressive_generate")

    # instantiate CLI (this will load models)
    print(f"Instantiating InferenceCLI(device={device}, target_model={target_model}, drafter_model={drafter_model}) ... (this will load your models)")
    cli = InferenceCLI(device=device, target_model=target_model, drafter_model=drafter_model)
    try:
        target_device = next(cli.target.parameters()).device
        drafter_device = next(cli.drafter.parameters()).device
        print(f"[INFO] Target model loaded on device: {target_device}")
        print(f"[INFO] Drafter model loaded on device: {drafter_device}")
    except Exception as e:
        print(f"[WARNING] Could not determine device for models: {e}")

    # override defaults with passed args
    cli.gen_len = gen_len
    cli.gamma = gamma

    # load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if k > len(dataset):
        print(f"[WARNING] Requested k={k} exceeds dataset size={len(dataset)}. Using full dataset instead.")
        k = len(dataset)
    # Sample k random indices upfront
    random.seed(seed)  # for reproducibility
    sample_indices = random.sample(range(len(dataset)), k)

    # load or create results file
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as rf:
                results_all = json.load(rf)
        except Exception:
            results_all = []
    else:
        results_all = []

    processed = len(results_all)
    if processed >= k:
        print(f"Already processed {processed} samples (>= requested k={k}). Exiting.")
        return

    print(f"Evaluating {k} samples, resuming from index {processed}. Results -> {output_path}")

    # helper: set seeds using CLI method (if available) else fallback
    def set_seed(s):
        if hasattr(cli, "_set_seed"):
            cli._set_seed(s)
        else:
            import random, numpy as np
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

    # iterate
    for count, idx in enumerate(tqdm(sample_indices[processed:], initial=processed, total=k, desc="Processing Samples"), start=processed):
        item = dataset[idx]
        qid = item.get("question_id", idx)
        prompt_list = item.get("prompt", [""])
        prompt_text = prompt_list[0] if isinstance(prompt_list, list) and len(prompt_list) > 0 else str(prompt_list)
        label = item.get("label", [])
        
        tokenized = prepare_tokens_from_cli(cli, prompt_text)

        entry: Dict[str, Any] = {
            "question_id": qid,
            "prompt": prompt_text,
            "label": label,
            "target_only": {},
            "drafter_only": {},
            "speculative": {},
        }

        # TARGET ONLY
        set_seed(seed)
        t0 = time.time()
        try:
            target_out_ids = autoregressive_generate(
                tokenized,
                cli.target,
                max_gen_len=cli.gen_len,
                eos_tokens_id=cli.end_tokens,
                logits_processor=cli.processor,
            )
        except Exception as e:
            print("Error during target-only generation:", e)
            target_out_ids = []
        t1 = time.time()
        target_text = cli.tokenizer.decode(target_out_ids, skip_special_tokens=True) if target_out_ids else ""
        target_tokens = len(target_out_ids)
        target_runtime = t1 - t0
        target_throughput = (target_tokens / target_runtime) if target_runtime > 0 else None

        entry["target_only"] = {
            "output_text": target_text,
            "tokens_generated": target_tokens,
            "runtime_sec": target_runtime,
            "throughput_toks_per_sec": target_throughput,
        }
        

        # DRAFTER ONLY
        set_seed(seed)
        t0 = time.time()
        try:
            drafter_out_ids = autoregressive_generate(
                tokenized,
                cli.drafter,
                max_gen_len=cli.gen_len,
                eos_tokens_id=cli.end_tokens,
                logits_processor=cli.processor,
            )
        except Exception as e:
            print("Error during drafter-only generation:", e)
            drafter_out_ids = []
        t1 = time.time()
        drafter_text = cli.tokenizer.decode(drafter_out_ids, skip_special_tokens=True) if drafter_out_ids else ""
        drafter_tokens = len(drafter_out_ids)
        drafter_runtime = t1 - t0
        drafter_throughput = (drafter_tokens / drafter_runtime) if drafter_runtime > 0 else None

        entry["drafter_only"] = {
            "output_text": drafter_text,
            "tokens_generated": drafter_tokens,
            "runtime_sec": drafter_runtime,
            "throughput_toks_per_sec": drafter_throughput,
        }
        

        # SPECULATIVE DECODING
        # wrap target model so we can measure time spent in target forward calls
        wrapped_target = TimedModelWrapper(cli.target)

        set_seed(seed)
        t_spec0 = time.time()
        try:
            spec_out_ids, accept_rate, reported_target_calls = speculative_generate(
                tokenized,
                cli.drafter,
                wrapped_target,
                tokenizer=cli.tokenizer,
                logits_processor=cli.processor,
                gamma=cli.gamma,
                max_gen_len=cli.gen_len,
                eos_tokens_id=cli.end_tokens,
            )
        except Exception as e:
            print("Error during speculative_generate:", e)
            spec_out_ids, accept_rate, reported_target_calls = [], 0.0, 0
        t_spec1 = time.time()

        spec_text = cli.tokenizer.decode(spec_out_ids, skip_special_tokens=True) if spec_out_ids else ""
        spec_tokens = len(spec_out_ids)
        spec_runtime = t_spec1 - t_spec0
        spec_throughput = (spec_tokens / spec_runtime) if spec_runtime > 0 else None

        entry["speculative"] = {
            "output_text": spec_text,
            "tokens_generated": spec_tokens,
            "runtime_sec": spec_runtime,
            "throughput_toks_per_sec": spec_throughput,
            "acceptance_rate": accept_rate,
            "spec_target_model_calls_reported": reported_target_calls,
            "spec_target_model_calls_wrapper_count": wrapped_target.calls,
            "spec_target_model_runtime_sec": wrapped_target.elapsed,
        }

        
        results_all.append(entry)
        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(results_all, wf, ensure_ascii=False, indent=2)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nCompleted {k} samples. Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speculative Decoding pipeline on cnnDailyMail dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to cnnDailyMail.json dataset")
    parser.add_argument("--specdec_path", type=str, required=True, help="Path to your specdec python file (defines InferenceCLI, speculative_generate, autoregressive_generate)")
    parser.add_argument("--k", type=int, default=10, help="Number of samples from start to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model loading (cuda or cpu or device_map string supported by your code)")
    parser.add_argument("--gen_len", type=int, default=64, help="Generation length for each sample")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma (number of drafts) for speculative decoding")
    parser.add_argument("--output", type=str, required=True, help="Output JSON results file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target_model", type=str, default=None,
                        help="HuggingFace repo id or local path for the TARGET model (overrides specdec defaults)")
    parser.add_argument("--drafter_model", type=str, default=None,
                        help="HuggingFace repo id or local path for the DRAFTER model (overrides specdec defaults)")

    args = parser.parse_args()

    evaluate(dataset_path=args.dataset,
             specdec_path=args.specdec_path,
             output_path=args.output,
             k=args.k,
             device=args.device,
             gen_len=args.gen_len,
             gamma=args.gamma,
             seed=args.seed,
             target_model=args.target_model,
             drafter_model=args.drafter_model,)


if __name__ == "__main__":
    main()




'''
#!/usr/bin/env python3
"""
evaluate_wmt.py
Uses specbench evaluation approach

Usage example:
python /home/iitb/Kishan_SpecDec/Archived/evaluate_WMT.py \
  --code-path "/home/iitb/Kishan_SpecDec/Archived/inference.py" \
  --wmt "/home/iitb/Kishan_SpecDec/Data/wmt.jsonl" \
  --out "/home/iitb/Kishan_SpecDec/resultsWMT/test1.json" \
  --summary "/home/iitb/Kishan_SpecDec/summaryWMT/wmtSummary.json" \
  --device "cuda:2" \
  --gamma 6 \
  --max-gen-len 64 \
  --max-examples 40 \
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

    # 4) load WMT dataset
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
    
    for i, rec in enumerate(tqdm(records[:max_examples], desc="Examples")):
        try:
            qid = rec.get("question_id", i)
            turns = rec.get("prompt", [])
            label = rec.get("label",[])
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
                "prompt_text": prompt_text,
                "label": label,
                
                "target_only": {
                    "output_text": ar_target_text,
                    "tokens_generated": ar_target_num_tokens,
                    "runtime_sec": ar_target_time,
                    "throughput_toks_per_sec": ar_target_throughput,
                },
                "baseline_draft": {
                    "output_text": ar_draft_text,
                    "tokens_generated": ar_draft_num_tokens,
                    "runtime_sec": ar_draft_time,
                    "throughput_toks_per_sec": ar_draft_throughput,
                },
                "speculative": {
                    "output_text": spec_text,
                    "tokens_generated": spec_num_tokens,
                    "runtime_sec": spec_total_time,
                    "throughput_toks_per_sec": spec_throughput,
                    "acceptance_rate": float(accept_rate) if accept_rate is not None else None,
                    "target_model_calls": measured_target_calls,
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
    parser.add_argument("--wmt", required=True, help="Path to WMT jsonl")
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
        specbench_path=args.wmt,
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


'''