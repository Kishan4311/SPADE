#!/usr/bin/env python3
"""
Usable:
    python /home/iitb/Kishan_SpecDec/utils/cnnDailyMail_summary.py

Reads input JSON from:
 "/home/iitb/Kishan_SpecDec/results_cnnDailyMail/cnnDailyMail_result_gm12_tgt8B_genLen128.json"

Writes summary JSON to:
 /home/iitb/Kishan_SpecDec/summary_cnnDailyMail/cnnDailyMail_summary_gm12_tgt8B_genLen128.json
"""

import json
from pathlib import Path
from statistics import mean, stdev
from typing import List
import math

# metric libs
try:
    import sacrebleu
except Exception as e:
    raise ImportError("Please install sacrebleu: pip install sacrebleu") from e

try:
    from rouge_score import rouge_scorer
except Exception as e:
    raise ImportError("Please install rouge-score: pip install rouge-score") from e

try:
    from pycocoevalcap.cider.cider import Cider
except Exception as e:
    raise ImportError(
        "Please install pycocoevalcap for CIDEr (pip install pycocoevalcap)."
    ) from e


INPUT_PATH = Path("/home/iitb/Kishan_SpecDec/resultsWMT/wmt_result_gm4_tgt3B_genLen128.json")
OUTPUT_PATH = Path("/home/iitb/Kishan_SpecDec/summaryWMT/summary_wmt_gm4_tgt3B_genLen128.json")


def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def extract_output_text(section: dict):
    """Return a string output_text from a model section (robust to malformed entries)."""
    if not section:
        return ""
    # prefer 'output_text' key
    if "output_text" in section:
        return safe_str(section.get("output_text", ""))
    # otherwise, find any string value
    for k, v in section.items():
        if isinstance(v, str):
            return v
    # fallback: stringify the whole section
    return safe_str(section)


def safe_get_number(section: dict, keys: List[str]):
    if not section:
        return None
    for k in keys:
        if k in section:
            try:
                return float(section[k])
            except Exception:
                # try cast from string
                try:
                    return float(str(section[k]))
                except Exception:
                    continue
    return None


def per_instance_bleu4(hyp: str, refs: List[str]) -> float:
    """
    Return BLEU-4 sentence score (0–100). Keeps case sensitivity.
    Uses sacrebleu with smoothing if available.
    """
    hyp = "" if hyp is None else str(hyp).strip()
    refs = [("" if r is None else str(r).strip()) for r in refs] or [""]
    if not hyp:
        return 0.0
    try:
        try:
            bleu = sacrebleu.sentence_bleu(hyp, refs, smooth_method="exp")
        except TypeError:
            bleu = sacrebleu.sentence_bleu(hyp, refs)
        return float(bleu.score)
    except Exception:
        return 0.0


def per_instance_bleu1(hyp: str, refs: List[str]) -> float:
    """
    Compute BLEU-1 (unigram) sentence score (0–100), case-sensitive.
    Simple unigram precision with brevity penalty.
    """
    hyp_text = "" if hyp is None else str(hyp).strip()
    refs_texts = [("" if r is None else str(r).strip()) for r in refs] or [""]

    if not hyp_text:
        return 0.0

    hyp_tokens = hyp_text.split()
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        return 0.0

    best_match = 0
    for r in refs_texts:
        ref_tokens = r.split()
        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        matched = 0
        seen = {}
        for t in hyp_tokens:
            if ref_counts.get(t, 0) - seen.get(t, 0) > 0:
                matched += 1
                seen[t] = seen.get(t, 0) + 1
        if matched > best_match:
            best_match = matched

    precision = best_match / hyp_len
    ref_lens = [len(r.split()) for r in refs_texts if r.strip() != ""]
    if not ref_lens:
        closest_ref_len = 0
    else:
        closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > closest_ref_len:
        bp = 1.0
    else:
        if closest_ref_len == 0:
            bp = 1.0 if precision > 0 else 0.0
        else:
            bp = math.exp(1 - (closest_ref_len / hyp_len))

    return bp * precision * 100.0



def per_instance_rouge_metric(hyp: str, refs: List[str], scorer=None, metric_name="rouge1") -> float:
    """
    Compute ROUGE (rouge1 or rougeL) F1 for hyp vs multiple refs (case-sensitive).
    Returns best F1 among refs ×100.
    """
    if scorer is None:
        scorer = rouge_scorer.RougeScorer([metric_name], use_stemmer=True)
    hyp = "" if hyp is None else str(hyp).strip()
    best_f1 = 0.0
    if not refs:
        refs = [""]
    for r in refs:
        r_text = "" if r is None else str(r).strip()
        try:
            sc = scorer.score(r_text, hyp)
            f1 = sc[metric_name].fmeasure
        except Exception:
            f1 = 0.0
        best_f1 = max(best_f1, f1)
    return best_f1 * 100.0


def compute_cider_scores_list(refs_list: List[List[str]], hyps: List[str]) -> List[float]:
    """
    Use pycocoevalcap Cider to compute per-instance CIDEr scores.
    Returns list of float scores in the same order.
    """
    # Build gts and res dicts with ids "0","1",...
    gts = {}
    res = {}
    for i, (refs, hyp) in enumerate(zip(refs_list, hyps)):
        id_str = str(i)
        # ensure we have at least one non-empty ref
        cleaned_refs = [safe_str(r) for r in refs if safe_str(r).strip()]
        if not cleaned_refs:
            cleaned_refs = [""]
        gts[id_str] = cleaned_refs
        res[id_str] = [safe_str(hyp)]
    cider = Cider()
    # cider.compute_score returns (mean_score, scores_list) where scores_list aligns with id order
    score, scores = cider.compute_score(gts, res)
    return [float(s)*10.0 for s in scores]


def safe_mean(lst):
    return float(mean(lst)) if lst else None


def safe_std(lst):
    # population or sample? we'll use sample stdev if len>1 else 0.0
    if not lst:
        return None
    if len(lst) == 1:
        return 0.0
    try:
        return float(stdev(lst))
    except Exception:
        return 0.0


def main():
    data = json.loads(INPUT_PATH.read_text())

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in input file.")

    # sort and find last question_id k
    data_sorted = sorted(data, key=lambda x: int(x.get("question_id", 0)))
    if not data_sorted:
        raise ValueError("No records found in input file.")
    k = max(int(rec.get("question_id", 0)) for rec in data_sorted)
    # select records with question_id in 1..k
    id_set = set(range(1, k + 1))
    records = [r for r in data_sorted if int(r.get("question_id", 0)) in id_set]
    records = sorted(records, key=lambda x: int(x.get("question_id", 0)))
    # Desired number of instances to use (set to None to use all)
    N = None

    # If N is None -> use all records, otherwise cap to available records.
    if N is None:
        N_used = len(records)
    else:
        if N <= 0:
            raise ValueError("N must be positive or None.")
        N_used = min(N, len(records))
        # slice records so downstream code processes only the first N_used items
        records = records[:N_used]
    print(f"Processing {N_used} records (of {len(data)} total).")
    if N_used == 0:
        raise ValueError("No records to process after applying filters / slicing.")

# N_used is the actual number of instances used

    # Prepare lists
    # BLEU & ROUGE lists
    bleu1_target_list, bleu1_drafter_list, bleu1_spec_list = [], [], []
    bleu4_target_list, bleu4_drafter_list, bleu4_spec_list = [], [], []
    rouge1_target_list, rouge1_drafter_list, rouge1_spec_list = [], [], []
    rougeL_target_list, rougeL_drafter_list, rougeL_spec_list = [], [], []

    # For each model, keep list of hyps, list of refs per instance
    refs_per_instance = []
    hyps_target = []
    hyps_drafter = []
    hyps_spec = []
    # numeric stats
    tgt_runtimes = []
    tgt_tokens = []
    tgt_throughputs = []

    dra_runtimes = []
    dra_tokens = []
    dra_throughputs = []

    spec_runtimes = []
    spec_tokens = []
    spec_throughputs = []

    spec_acceptances = []
    spec_calls = []
    spec_target_runtimes = []

    for rec in records:
        # labels: list of strings usually
        labels = rec.get("label", [])
        # sanitize labels
        refs = [safe_str(l).strip() for l in labels if safe_str(l).strip()]
        if not refs:
            print(f"Warning: Empty references for question_id {rec.get('question_id')}. Using empty string as fallback.")
            refs = [""]
        refs_per_instance.append(refs)

        # outputs
        hyps_target.append(extract_output_text(rec.get("target_only", {})).strip())
        hyps_drafter.append(extract_output_text(rec.get("drafter_only", {})).strip())
        hyps_spec.append(extract_output_text(rec.get("speculative", {})).strip())

        # collect numeric stats robustly
        # target_only
        tsec = safe_get_number(rec.get("target_only", {}), ["runtime_sec", "runtime_secs", "runtime"])
        if tsec is not None:
            tgt_runtimes.append(tsec)
        ttok = safe_get_number(rec.get("target_only", {}), ["tokens_generated", "tokens", "token_generated"])
        if ttok is not None:
            tgt_tokens.append(ttok)
        tthr = safe_get_number(rec.get("target_only", {}), ["throughput_toks_per_sec", "throughput", "toks_per_sec"])
        if tthr is not None:
            tgt_throughputs.append(tthr)

        # drafter_only
        dsec = safe_get_number(rec.get("drafter_only", {}), ["runtime_sec", "runtime_secs", "runtime"])
        if dsec is not None:
            dra_runtimes.append(dsec)
        dtok = safe_get_number(rec.get("drafter_only", {}), ["tokens_generated", "tokens", "token_generated"])
        if dtok is not None:
            dra_tokens.append(dtok)
        dthr = safe_get_number(rec.get("drafter_only", {}), ["throughput_toks_per_sec", "throughput", "toks_per_sec"])
        if dthr is not None:
            dra_throughputs.append(dthr)

        # speculative
        ssec = safe_get_number(rec.get("speculative", {}), ["runtime_sec", "runtime_secs", "runtime"])
        if ssec is not None:
            spec_runtimes.append(ssec)
        stok = safe_get_number(rec.get("speculative", {}), ["tokens_generated", "tokens", "token_generated"])
        if stok is not None:
            spec_tokens.append(stok)
        sthr = safe_get_number(rec.get("speculative", {}), ["throughput_toks_per_sec", "throughput", "toks_per_sec"])
        if sthr is not None:
            spec_throughputs.append(sthr)

        # speculative-specific
        sacc = safe_get_number(rec.get("speculative", {}), ["acceptance_rate", "accept_rate"])
        if sacc is not None:
            spec_acceptances.append(sacc)
        scalls = safe_get_number(rec.get("speculative", {}), ["spec_target_model_calls_reported",
                                                            "spec_target_model_calls_wrapper_count",
                                                            "spec_target_model_calls"])
        if scalls is not None:
            spec_calls.append(scalls)
        s_t_rt = safe_get_number(rec.get("speculative", {}), ["spec_target_model_runtime_sec", "spec_target_model_runtime"])
        if s_t_rt is not None:
            spec_target_runtimes.append(s_t_rt)

    # Compute per-instance BLEU and ROUGE for each model
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    # compute BLEU and ROUGE per instance
    for refs, ht, hd, hs in zip(refs_per_instance, hyps_target, hyps_drafter, hyps_spec):
        # BLEU-4
        b4_t = per_instance_bleu4(ht, refs)
        b4_d = per_instance_bleu4(hd, refs)
        b4_s = per_instance_bleu4(hs, refs)
        bleu4_target_list.append(b4_t)
        bleu4_drafter_list.append(b4_d)
        bleu4_spec_list.append(b4_s)

        # BLEU-1
        b1_t = per_instance_bleu1(ht, refs)
        b1_d = per_instance_bleu1(hd, refs)
        b1_s = per_instance_bleu1(hs, refs)
        bleu1_target_list.append(b1_t)
        bleu1_drafter_list.append(b1_d)
        bleu1_spec_list.append(b1_s)

        # ROUGE-1
        r1_t = per_instance_rouge_metric(ht, refs, scorer=rouge_scorer_obj, metric_name="rouge1")
        r1_d = per_instance_rouge_metric(hd, refs, scorer=rouge_scorer_obj, metric_name="rouge1")
        r1_s = per_instance_rouge_metric(hs, refs, scorer=rouge_scorer_obj, metric_name="rouge1")
        rouge1_target_list.append(r1_t)
        rouge1_drafter_list.append(r1_d)
        rouge1_spec_list.append(r1_s)

        # ROUGE-L
        rL_t = per_instance_rouge_metric(ht, refs, scorer=rouge_scorer_obj, metric_name="rougeL")
        rL_d = per_instance_rouge_metric(hd, refs, scorer=rouge_scorer_obj, metric_name="rougeL")
        rL_s = per_instance_rouge_metric(hs, refs, scorer=rouge_scorer_obj, metric_name="rougeL")
        rougeL_target_list.append(rL_t)
        rougeL_drafter_list.append(rL_d)
        rougeL_spec_list.append(rL_s)

    # Compute CIDEr per-instance for each model using Cider (do this AFTER per-instance hyps/ref lists are ready)
    cider_target_list = compute_cider_scores_list(refs_per_instance, hyps_target)
    cider_drafter_list = compute_cider_scores_list(refs_per_instance, hyps_drafter)
    cider_spec_list = compute_cider_scores_list(refs_per_instance, hyps_spec)

    # Aggregate stats: mean and std for each metric and requested numeric stats
    summary = {}

    summary["Target_only"] = {
        "bleu1_mean": safe_mean(bleu1_target_list),
        "bleu1_std": safe_std(bleu1_target_list),
        "bleu4_mean": safe_mean(bleu4_target_list),
        "bleu4_std": safe_std(bleu4_target_list),
        "rouge1_f1_mean": safe_mean(rouge1_target_list),
        "rouge1_f1_std": safe_std(rouge1_target_list),
        "rougeL_f1_mean": safe_mean(rougeL_target_list),
        "rougeL_f1_std": safe_std(rougeL_target_list),
        "ciderD_mean": safe_mean(cider_target_list),
        "ciderD_std": safe_std(cider_target_list),
        "mean_runtime_sec": safe_mean(tgt_runtimes),
        "mean_tokens_generated": safe_mean(tgt_tokens),
        "mean_throughput_toks_per_sec": safe_mean(tgt_throughputs),
    }

    summary["Drafter_only"] = {
        "bleu1_mean": safe_mean(bleu1_drafter_list),
        "bleu1_std": safe_std(bleu1_drafter_list),
        "bleu4_mean": safe_mean(bleu4_drafter_list),
        "bleu4_std": safe_std(bleu4_drafter_list),

        "rouge1_f1_mean": safe_mean(rouge1_drafter_list),
        "rouge1_f1_std": safe_std(rouge1_drafter_list),
        "rougeL_f1_mean": safe_mean(rougeL_drafter_list),
        "rougeL_f1_std": safe_std(rougeL_drafter_list),

        "ciderD_mean": safe_mean(cider_drafter_list),
        "ciderD_std": safe_std(cider_drafter_list),

        "mean_runtime_sec": safe_mean(dra_runtimes),
        "mean_tokens_generated": safe_mean(dra_tokens),
        "mean_throughput_toks_per_sec": safe_mean(dra_throughputs),
    }

    summary["Speculative"] = {
        "bleu1_mean": safe_mean(bleu1_spec_list),
        "bleu1_std": safe_std(bleu1_spec_list),
        "bleu4_mean": safe_mean(bleu4_spec_list),
        "bleu4_std": safe_std(bleu4_spec_list),

        "rouge1_f1_mean": safe_mean(rouge1_spec_list),
        "rouge1_f1_std": safe_std(rouge1_spec_list),
        "rougeL_f1_mean": safe_mean(rougeL_spec_list),
        "rougeL_f1_std": safe_std(rougeL_spec_list),

        "ciderD_mean": safe_mean(cider_spec_list),
        "ciderD_std": safe_std(cider_spec_list),

        "mean_runtime_sec": safe_mean(spec_runtimes),
        "mean_tokens_generated": safe_mean(spec_tokens),
        "mean_spec_target_model_calls": safe_mean(spec_calls),
        "mean_throughput_toks_per_sec": safe_mean(spec_throughputs),
        "mean_acceptance_rate": safe_mean(spec_acceptances),
    }

    # SpecDec target model (target model runtime inside specdec): mean runtime
    summary["SpecDec_target_model_in_spec"] = {
        "mean_runtime_sec": safe_mean(spec_target_runtimes)
    }

    # Add meta
    summary["_meta"] = {
        "input_path": str(INPUT_PATH),
        "instances_used": N_used,
        "k_last_question_id": k
    }

    # Write JSON summary
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
