import json
import re
from pathlib import Path

# Paths (adjust if needed)
QUESTION_FILE = Path("data/SpecBench_question.jsonl")
RESULTS_FILE = Path("/home/iitb/Kishan_SpecDec/results_specbench/specbench_results_gm8_tgt8B_genLen256.jsonl")
OUTPUT_FILE = Path("/home/iitb/Kishan_SpecDec/results_specbench/specbench_results_gm8_tgt8B_genLen256_cleaned.jsonl")

# categories to treat as "multi_turn_conversation" 
MULTI_TURN_CATS = {"writing", "math", "coding", "reasoning", "roleplay", "extraction","stem", "humanities"}


# Regex to remove model token markers like <|begin_of_text|> and any <|...|>
MODEL_TOKEN_RE = re.compile(r"<\|.*?\|>")

# Regex to collapse multiple whitespace/newlines into single spaces and remove repeated blank lines
WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text):
    """
    Clean returned/ prompt text:
    - Remove model special tokens of the form <|...|>
    - Remove backslashes and escaped sequences like \n, \t
    - Collapse repeated whitespace/newlines into single spaces
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    # Remove explicit model tokens like <|...|>
    text = MODEL_TOKEN_RE.sub(" ", text)
    # Remove common escaped sequences and stray backslashes
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace("\\", " ")
    # Replace any remaining control characters
    text = "".join(ch for ch in text if ch >= " " or ch == "\n")
    # Collapse whitespace (including newlines) into single spaces
    text = WHITESPACE_RE.sub(" ", text)
    # Final strip
    return text.strip()

#  token-count extractor 
def extract_token_count(obj):
    """
    Try to find a token-count integer in `obj` (a dict). Returns int or None.
    Checks a list of common keys and also inspects nested dicts like 'stats' or 'metadata'.
    """
    if not isinstance(obj, dict):
        return None

    # common top-level keys that might contain token counts
    candidate_keys = [
        "num_generated_tokens", "generated_tokens", "num_tokens",
        "token_count", "n_tokens_generated", "tokens_generated",
        "gen_tokens", "n_generated_tokens", "generated_token_count",
        "num_output_tokens", "output_token_count", "num_gen_tokens"
    ]

    for k in candidate_keys:
        v = obj.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, (str, float)):
            try:
                return int(v)
            except Exception:
                pass

    # check common nested places
    nested_parents = ["stats", "metadata", "generation_info", "meta", "generation_stats"]
    for parent in nested_parents:
        nested = obj.get(parent)
        if isinstance(nested, dict):
            for k in candidate_keys:
                v = nested.get(k)
                if isinstance(v, int):
                    return v
                if isinstance(v, (str, float)):
                    try:
                        return int(v)
                    except Exception:
                        pass

    # if nothing found, return None
    return None

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
            prompt = " ".join(str(t) for t in turns if t is not None)
        else:
            # fallback to any prompt-like fields
            prompt = obj.get("prompt_text") or obj.get("prompt") or ""
        questions[qid] = clean_text(prompt)

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

        target_out = baseline_target.get("output_text") or baseline_target.get("text") or ""
        draft_out = baseline_draft.get("output_text") or baseline_draft.get("text") or ""
        spec_out = speculative.get("output_text") or speculative.get("text") or ""

        # extract token counts (may be None if not present) 
        target_tokens = extract_token_count(baseline_target)
        draft_tokens  = extract_token_count(baseline_draft)
        spec_tokens   = extract_token_count(speculative)
        # 

        results[qid] = {
            "targetonlyoutput": clean_text(target_out),
            "draftonlyoutput": clean_text(draft_out),
            "specbenchoutput": clean_text(spec_out),
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
    # 

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
