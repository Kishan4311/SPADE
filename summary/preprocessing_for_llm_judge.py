#!/usr/bin/env python3
"""
Usable:

python /summary/preprocessing_for_llm_judge.py \
    --questions data/SpecBench_question.jsonl \
    --results /home/iitb/Kishan_SpecDec/results_specbench/specbench_results_gm8_tgt8B_genLen256.jsonl \
    --out /home/iitb/Kishan_SpecDec/Archived/specbench_results_gm8_tgt8B_genLen256_preprocessed.jsonl

"""
This script preprocesses SpecBench results by merging question prompts and model outputs.  
It reads question and result JSONL files, aligns them by question_id, cleans the data,  
and writes a consolidated JSONL file ready for evaluation by Judge.
"""

"""

import json
import argparse
from pathlib import Path

# categories to treat as "multi_turn_conversation"
MULTI_TURN_CATS = {"writing", "math", "coding", "reasoning", "roleplay", "extraction", "stem", "humanities"}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SpecBench results. Provide paths for questions, results, and output."
    )
    parser.add_argument(
        "--questions", "-q", required=True, type=Path,
        help="Path to SpecBench question jsonl (e.g. /path/to/SpecBench_question.jsonl)"
    )
    parser.add_argument(
        "--results", "-r", required=True, type=Path,
        help="Path to per-prompt results jsonl (e.g. /path/to/specbench_results.jsonl)"
    )
    parser.add_argument(
        "--out", "-o", required=True, type=Path,
        help="Path to write preprocessed jsonl (e.g. /path/to/preprocessed.jsonl)"
    )

    args = parser.parse_args()

    QUESTION_FILE = args.questions
    RESULTS_FILE = args.results
    OUTPUT_FILE = args.out

    # Load questions -> map question_id to prompt_text (join turns if list present)
    questions = {}
    with QUESTION_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("question_id")
            if qid is None:
                continue
            # 'turns' is a list of prompt pieces; join them with a single space
            turns = obj.get("turns")
            if isinstance(turns, list):
                prompt = "\n\n".join(str(t) for t in turns if t is not None)
            else:
                # fallback to any prompt-like fields
                prompt = obj.get("prompt_text") or obj.get("prompt") or ""
            questions[qid] = prompt

    # Load results -> map question_id to cleaned outputs (only output_text fields) + token counts
    results = {}
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("question_id")
            if qid is None:
                continue

            # Safely extract nested fields; some entries may omit them
            baseline_target = obj.get("baseline_target", {}) or {}
            baseline_draft = obj.get("baseline_draft", {}) or {}
            speculative = obj.get("speculative", {}) or {}

            target_out = baseline_target.get("output_text") or ""
            draft_out = baseline_draft.get("output_text") or ""
            spec_out = speculative.get("output_text") or ""

            target_tokens = baseline_target.get("num_tokens") or ""
            draft_tokens = baseline_draft.get("num_tokens") or ""
            spec_tokens = speculative.get("num_tokens") or ""

            results[qid] = {
                "targetonlyoutput": target_out,
                "draftonlyoutput": draft_out,
                "specbenchoutput": spec_out,
                "category": obj.get("category", ""),
                "target_token_count": target_tokens,
                "draft_token_count": draft_tokens,
                "spec_token_count": spec_tokens
            }

    # Merge and write cleaned array (JSONL)
    cleaned = []
    for qid, prompt in questions.items():
        # Determine category (prefer results' category if present, else default to "writing")
        raw_category = results.get(qid, {}).get("category") or "writing"

        # --- map certain categories to "multi_turn_conversation" ---
        if raw_category in MULTI_TURN_CATS:
            final_category = "multi_turn_conversation"
        else:
            final_category = raw_category

        if qid not in results:
            entry = {
                "question_id": qid,
                "category": final_category,
                "prompt_text": prompt,
                "targetonlyoutput": "",
                "draftonlyoutput": "",
                "specbenchoutput": "",
                "target_token_count": None,
                "draft_token_count": None,
                "spec_token_count": None
            }
        else:
            entry = {
                "question_id": qid,
                "category": final_category,
                "prompt_text": prompt,
                "targetonlyoutput": results[qid]["targetonlyoutput"],
                "draftonlyoutput": results[qid]["draftonlyoutput"],
                "specbenchoutput": results[qid]["specbenchoutput"],
                "target_token_count": results[qid].get("target_token_count"),
                "draft_token_count": results[qid].get("draft_token_count"),
                "spec_token_count": results[qid].get("spec_token_count")
            }
        cleaned.append(entry)

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        # writing JSONL style: one JSON object per line
        for obj in cleaned:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(cleaned)} entries to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
