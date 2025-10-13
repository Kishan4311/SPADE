#!/usr/bin/env python3
"""
evaluate_cnnDailyMail.py

Usage example:
python evaluate_cnnDailyMail.py \
--device "cuda:3" \
  --dataset "/home/iitb/Kishan_SpecDec/Data/cnnDailyMail .json" \
  --specdec_path "/home/iitb/Kishan_SpecDec/llama_3_2_1b_specdec_v2 (1).py" \
  --k 200 \
  --gen_len 128 \
  --gamma 12 \
  --target_model "meta-llama/Llama-3.1-8B-Instruct" \
  --drafter_model "meta-llama/Llama-3.2-1B-Instruct" \
  --output "/home/iitb/Kishan_SpecDec/results_cnnDailyMail/cnnDailyMail_result_gm12_tgt8B_genLen128.json"


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
    """
    Wraps a HuggingFace model so each call is timed and counted.
    Delegates attributes to underlying model.
    """
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
    Fail gracefully if tokenizer.apply_chat_template is missing.
    """
    try:
        if getattr(cli, "chat", False):
            # some tokenizers support apply_chat_template as in user's notebook
            prompt_wrap = cli.tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
        else:
            prompt_wrap = prompt_text
    except Exception:
        # fallback
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

    # instantiate CLI (this will load models )
    print(f"Instantiating InferenceCLI(device={device}, target_model={target_model}, drafter_model={drafter_model}) ... (this will load your models)")
    cli = InferenceCLI(device=device, target_model=target_model, drafter_model=drafter_model)
    # cli = InferenceCLI(device=device)
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

    # set seeds using CLI method (if available) else fallback
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
                use_cache=cli.cache,
                max_gen_len=cli.gen_len,
                eos_tokens_id=cli.end_tokens,
                logits_processor=cli.processor,
                debug=False,
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
                use_cache=cli.cache,
                max_gen_len=cli.gen_len,
                eos_tokens_id=cli.end_tokens,
                logits_processor=cli.processor,
                debug=False,
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
                debug=False,
                use_cache=cli.cache,
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
        # write incrementally
        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(results_all, wf, ensure_ascii=False, indent=2)

        # try freeing GPU cache (optional)
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
